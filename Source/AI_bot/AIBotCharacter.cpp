#include "AIBotCharacter.h"
#include "NNE.h"
#include "NNERuntimeGPU.h"

AAIBotCharacter::AAIBotCharacter() { PrimaryActorTick.bCanEverTick = true; }

void AAIBotCharacter::BeginPlay()
{
    Super::BeginPlay();
    TWeakInterfacePtr<INNERuntimeGPU> Runtime = UE::NNE::GetRuntime<INNERuntimeGPU>(FString("NNERuntimeORTDml"));
    if (!Runtime.IsValid()) return;

    // モデル1: AIM (既存 3次元)
    if (ModelData) {
        GPUModel = Runtime->CreateModelGPU(ModelData);
        if (GPUModel) {
            ModelInstance = GPUModel->CreateModelInstanceGPU();
            if (ModelInstance) {
                TArray<UE::NNE::FTensorShape> Shapes;
                Shapes.Add(UE::NNE::FTensorShape::Make({ 1, 3 }));
                ModelInstance->SetInputTensorShapes(Shapes);
            }
        }
    }

    // モデル2: PEAK (最新 22次元)
    if (ModelData2) {
        GPUModel2 = Runtime->CreateModelGPU(ModelData2);
        if (GPUModel2) {
            ModelInstancePeak = GPUModel2->CreateModelInstanceGPU();
            if (ModelInstancePeak) {
                TArray<UE::NNE::FTensorShape> Shapes;
                // 22次元にアップデート
                Shapes.Add(UE::NNE::FTensorShape::Make({ 1, 22 }));
                ModelInstancePeak->SetInputTensorShapes(Shapes);
            }
        }
    }
}

void AAIBotCharacter::RunInference(float RelPitch, float RelYaw, float Distance, float& OutTurn, float& OutLockup, float& bOutShouldFire)
{
    if (!ModelInstance.IsValid()) return;
    TArray<float> Input = { RelPitch / 45.0f, RelYaw / 180.0f, Distance / 3000.0f };
    TArray<float> Output = { 0.0f, 0.0f, 0.0f };
    TArray<UE::NNE::FTensorBindingCPU> Inputs, Outputs;
    Inputs.Add({ Input.GetData(), (uint64)Input.Num() * sizeof(float) });
    Outputs.Add({ Output.GetData(), (uint64)Output.Num() * sizeof(float) });
    ModelInstance->RunSync(Inputs, Outputs);
    OutTurn = Output[0] * 12.0f; OutLockup = Output[1] * 3.0f; bOutShouldFire = Output[2];
}

void AAIBotCharacter::RunInferencePeak(
    const TArray<float>& Rays, const TArray<float>& WallDist45, const TArray<float>& WallDist90,
    bool bIsTargetVisible, float TargetDistance, FVector MyVelocity, FVector TargetVelocity,
    float TimeTargetVisible, FVector2D CurrentAimError, float MyPitch,
    float& OutMoveRight, float& OutMoveForward, float& OutFireValue, float& OutTurnDelta, float& OutLookUpDelta
)
{
    if (!ModelInstancePeak.IsValid()) return;

    // --- 1. 22次元入力構築 ---
    TArray<float> InputBuffer;
    InputBuffer.Reserve(22);

    for (int32 i = 0; i < 7; ++i) { InputBuffer.Add(FMath::Clamp(Rays[i] / 5000.0f, 0.0f, 1.0f)); }
    InputBuffer.Add(FMath::Clamp(WallDist45[0] / 500.0f, 0.0f, 1.0f));
    InputBuffer.Add(FMath::Clamp(WallDist45[1] / 500.0f, 0.0f, 1.0f));
    InputBuffer.Add(FMath::Clamp(WallDist90[0] / 500.0f, 0.0f, 1.0f));
    InputBuffer.Add(FMath::Clamp(WallDist90[1] / 500.0f, 0.0f, 1.0f));
    InputBuffer.Add(bIsTargetVisible ? 1.0f : 0.0f);
    InputBuffer.Add(FMath::Clamp(TargetDistance / 5000.0f, 0.0f, 1.0f));
    // 自己速度 (13-14)
    InputBuffer.Add(FMath::Clamp(MyVelocity.X / 600.0f, -1.0f, 1.0f));
    InputBuffer.Add(FMath::Clamp(MyVelocity.Y / 600.0f, -1.0f, 1.0f));
    // 敵速度 (15-17)
    InputBuffer.Add(FMath::Clamp(TargetVelocity.X / 600.0f, -1.0f, 1.0f));
    InputBuffer.Add(FMath::Clamp(TargetVelocity.Y / 600.0f, -1.0f, 1.0f));
    InputBuffer.Add(FMath::Clamp(TargetVelocity.Z / 600.0f, -1.0f, 1.0f));
    // その他 (18-21)
    InputBuffer.Add(FMath::Clamp(TimeTargetVisible / 2000.0f, 0.0f, 1.0f));
    InputBuffer.Add(FMath::Clamp(CurrentAimError.X / 100.0f, -1.0f, 1.0f));
    InputBuffer.Add(FMath::Clamp(CurrentAimError.Y / 100.0f, -1.0f, 1.0f));
    InputBuffer.Add(FMath::Clamp(MyPitch / 90.0f, -1.0f, 1.0f));

    // --- 2. 出力 ---
    TArray<float> OutMoveR = { 0.0f, 0.0f, 0.0f };
    TArray<float> OutMoveF = { 0.0f, 0.0f, 0.0f };
    float OutFire = 0.0f;
    TArray<float> OutAim = { 0.0f, 0.0f };

    TArray<UE::NNE::FTensorBindingCPU> InBindings, OutBindings;
    InBindings.Add({ InputBuffer.GetData(), (uint64)InputBuffer.Num() * sizeof(float) });
    OutBindings.Add({ OutMoveR.GetData(), (uint64)OutMoveR.Num() * sizeof(float) });
    OutBindings.Add({ OutMoveF.GetData(), (uint64)OutMoveF.Num() * sizeof(float) });
    OutBindings.Add({ &OutFire, sizeof(float) });
    OutBindings.Add({ OutAim.GetData(), (uint64)OutAim.Num() * sizeof(float) });

    ModelInstancePeak->RunSync(InBindings, OutBindings);

    auto GetCls = [](const TArray<float>& B) {
        int32 Mi = 0; float Mv = -999.f;
        for (int32 i = 0; i < 3; ++i) { if (B[i] > Mv) { Mv = B[i]; Mi = i; } }
        return (float)(Mi - 1);
        };
    OutMoveRight = GetCls(OutMoveR); OutMoveForward = GetCls(OutMoveF); OutFireValue = OutFire;
    float Sens = 5.0f; OutTurnDelta = OutAim[0] * Sens; OutLookUpDelta = OutAim[1] * Sens;
}

void AAIBotCharacter::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);
}

void AAIBotCharacter::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
    Super::SetupPlayerInputComponent(PlayerInputComponent);
}