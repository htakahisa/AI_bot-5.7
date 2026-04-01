#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "NNEModelData.h"
#include "NNERuntimeGPU.h"
#include "NNEManagerComponent.generated.h"

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class AI_BOT_API UNNEManagerComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UNNEManagerComponent();

    UFUNCTION(BlueprintCallable, Category = "NNE")
    bool LoadModel(UNNEModelData* ModelData);

    UFUNCTION(BlueprintCallable, Category = "NNE")
    TArray<float> RunInference(const TArray<float>& InputData);

    UFUNCTION(BlueprintCallable, Category = "NNE")
    void ReleaseModel();

private:
    TSharedPtr<UE::NNE::IModelInstanceGPU> ModelInstance;
};