#include "AIBotCharacter.h"
#include "NNE.h"
#include "NNERuntimeGPU.h"

AAIBotCharacter::AAIBotCharacter()
{
    PrimaryActorTick.bCanEverTick = true;
}

void AAIBotCharacter::BeginPlay()
{
    Super::BeginPlay();

    // UE5.7 DirectML Runtimeの取得
    TWeakInterfacePtr<INNERuntimeGPU> Runtime =
        UE::NNE::GetRuntime<INNERuntimeGPU>(FString("NNERuntimeORTDml"));

    if (!Runtime.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("NNE: GPU Runtime is NULL!"));
        return;
    }

    // ── モデル1：AIM（既存仕様：Input:3, Output:3） ──
    if (ModelData)
    {
        GPUModel = Runtime->CreateModelGPU(ModelData);
        if (GPUModel)
        {
            ModelInstance = GPUModel->CreateModelInstanceGPU();
            if (ModelInstance)
            {
                TArray<UE::NNE::FTensorShape> Shapes;
                Shapes.Add(UE::NNE::FTensorShape::Make({ 1, 3 }));
                ModelInstance->SetInputTensorShapes(Shapes);
                UE_LOG(LogTemp, Log, TEXT("Model AIM: Initialized OK"));
            }
        }
    }

    // ── モデル2：PEAK（新仕様：Input:21, Output:5） ──
    if (ModelData2)
    {
        GPUModel2 = Runtime->CreateModelGPU(ModelData2);
        if (GPUModel2)
        {
            ModelInstancePeak = GPUModel2->CreateModelInstanceGPU();
            if (ModelInstancePeak)
            {
                TArray<UE::NNE::FTensorShape> Shapes;
                // 代表の最新リストに基づき、Input 21次元を設定
                Shapes.Add(UE::NNE::FTensorShape::Make({ 1, 21 }));
                ModelInstancePeak->SetInputTensorShapes(Shapes);
                UE_LOG(LogTemp, Log, TEXT("Model PEAK: Initialized with 21 inputs OK"));
            }
        }
    }
}

/**
 * モデル1: AIM 推論（input:3, output:3）
 * 敵との相対角度と距離から、レティクル操作と射撃判断を行う。
 */
void AAIBotCharacter::RunInference(
    float RelPitch,
    float RelYaw,
    float Distance,
    float& OutTurn,
    float& OutLockup,
    float& bOutShouldFire
)
{
    if (!ModelInstance.IsValid()) return;

    // 学習時の正規化係数に合わせて入力
    TArray<float> Input = {
        RelPitch / 40.0f,
        RelYaw / 120.0f,
        Distance / 3600.0f
    };
    TArray<float> Output = { 0.0f, 0.0f, 0.0f };

    TArray<UE::NNE::FTensorBindingCPU> Inputs, Outputs;
    Inputs.Add({ Input.GetData(),  (uint64)Input.Num() * sizeof(float) });
    Outputs.Add({ Output.GetData(), (uint64)Output.Num() * sizeof(float) });

    ModelInstance->RunSync(Inputs, Outputs);

    // 出力スケーリング
    OutTurn = Output[0] * 12.0f;
    OutLockup = Output[1] * 3.0f;
    bOutShouldFire = Output[2];
}

/**
 * モデル2: PEAK 推論（input:21, output:5）
 * レイ7本、自機速度、Pitch等を含む21次元データから、足回りと視点移動を制御。
 */
void AAIBotCharacter::RunInferencePeak(
    float Ray0, float Ray1, float Ray2, float Ray3, float Ray4, float Ray5, float Ray6,
    float DistToWall,
    bool bIsTargetVisible,
    float TimeTargetVisible,
    float TargetDistance,
    float TargetVelocity,
    float CurrentAimError,
    float MyVelocity,
    bool bIsReloading,
    float CurrentMoveRight,
    float CurrentMoveForward,
    bool bIsStoppingTrigger,
    bool bIsFireIn,
    bool bIsCrouching,
    float MyPitch,
    float& OutMoveRight, float& OutMoveForward, float& OutFireValue, float& OutTurnDelta, float& OutLookUpDelta
)
{
    if (!ModelInstancePeak.IsValid()) return;

    // --- 1. 21次元の入力配列を構築 (正規化込み) ---
    TArray<float> Inputs;
    Inputs.Reserve(21);

    // 1-7. distances
    Inputs.Add(Ray0 / 5000.0f); Inputs.Add(Ray1 / 5000.0f); Inputs.Add(Ray2 / 5000.0f);
    Inputs.Add(Ray3 / 5000.0f); Inputs.Add(Ray4 / 5000.0f); Inputs.Add(Ray5 / 5000.0f);
    Inputs.Add(Ray6 / 5000.0f);

    // 8. distToWall
    Inputs.Add(DistToWall / 5000.0f);
    // 9. isTargetVisible
    Inputs.Add(bIsTargetVisible ? 1.0f : 0.0f);
    // 10. timeTargetVisible (最大2秒)
    Inputs.Add(FMath::Clamp(TimeTargetVisible / 2.0f, 0.0f, 1.0f));
    // 11. targetDistance
    Inputs.Add(TargetDistance / 5000.0f);
    // 12. targetVelocity
    Inputs.Add(TargetVelocity / 600.0f);
    // 13. currentAimError
    Inputs.Add(CurrentAimError / 180.0f);
    // 14. myVelocity
    Inputs.Add(MyVelocity / 600.0f);
    // 15. isReloading
    Inputs.Add(bIsReloading ? 1.0f : 0.0f);
    // 16-17. currentMoveInput
    Inputs.Add(CurrentMoveRight);
    Inputs.Add(CurrentMoveForward);
    // 18. isStoppingTrigger
    Inputs.Add(bIsStoppingTrigger ? 1.0f : 0.0f);
    // 19. isFireIn
    Inputs.Add(bIsFireIn ? 1.0f : 0.0f);
    // 20. isCrouching
    Inputs.Add(bIsCrouching ? 1.0f : 0.0f);
    // 21. myPitch
    Inputs.Add(MyPitch / 90.0f);

    // --- 2. 推論実行 ---
    TArray<float> OutputBuffer = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    TArray<UE::NNE::FTensorBindingCPU> InBindings, OutBindings;
    InBindings.Add({ (void*)Inputs.GetData(), (uint64)Inputs.Num() * sizeof(float) });
    OutBindings.Add({ OutputBuffer.GetData(), (uint64)OutputBuffer.Num() * sizeof(float) });

    ModelInstancePeak->RunSync(InBindings, OutBindings);

    // --- 3. 出力の割り当て（RunInferencePeakの最後） ---

    // 移動のスケーリング（先ほどのBoost）
    auto Digitize = [](float InValue) -> float {
        if (InValue > 0.05f) return 1.0f;
        if (InValue < -0.05f) return -1.0f;
        return 0.0f;
        };
    OutMoveRight = Digitize(OutputBuffer[0]);
    OutMoveForward = Digitize(OutputBuffer[1]);
    OutFireValue = OutputBuffer[2];

    // --- 視点移動の増幅 ---
    // 視点移動は移動量（Delta）なので、大きな係数を掛けないと変化が見えません
    // 1.0の出力があった時に「5度」動かす設定例（感度調整）
    float MouseSensitivity = 5.0f;
    OutTurnDelta = OutputBuffer[3] * MouseSensitivity;
    OutLookUpDelta = OutputBuffer[4] * MouseSensitivity;
}

void AAIBotCharacter::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);
}

void AAIBotCharacter::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
    Super::SetupPlayerInputComponent(PlayerInputComponent);
}