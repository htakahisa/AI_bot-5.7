#pragma once
#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "NNEModelData.h"
#include "NNE.h"
#include "NNERuntimeGPU.h"
#include "AIBotCharacter.generated.h"

UCLASS()
class AI_BOT_API AAIBotCharacter : public ACharacter
{
    GENERATED_BODY()

public:
    AAIBotCharacter();

    // モデル1（input:3, output:3）
    UPROPERTY(EditAnywhere, Category = "NNE")
    TObjectPtr<UNNEModelData> ModelData;

    // モデル2（input:10, output:3）
    UPROPERTY(EditAnywhere, Category = "NNE")
    TObjectPtr<UNNEModelData> ModelData2;

    // モデル1用推論
    UFUNCTION(BlueprintCallable, Category = "NNE")
    void RunInference(float RelPitch, float RelYaw, float Distance,
        float& OutTurn, float& OutLockup, float& bOutShouldFire);

    // モデル2用推論（input10個、output3個）
    UFUNCTION(BlueprintCallable, Category = "NNE")
    void RunInferencePeak(
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
    );

protected:
    virtual void BeginPlay() override;

public:
    virtual void Tick(float DeltaTime) override;
    virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;

private:
    // モデル1
    TSharedPtr<UE::NNE::IModelGPU> GPUModel;
    TSharedPtr<UE::NNE::IModelInstanceGPU> ModelInstance;

    // モデル2
    TSharedPtr<UE::NNE::IModelGPU> GPUModel2;
    TSharedPtr<UE::NNE::IModelInstanceGPU> ModelInstancePeak;
};