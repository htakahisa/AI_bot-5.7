#include "NNEManagerComponent.h"
#include "NNE.h"
#include "NNERuntimeGPU.h"
#include "NNEModelData.h"

UNNEManagerComponent::UNNEManagerComponent()
{
    PrimaryComponentTick.bCanEverTick = false;
}

bool UNNEManagerComponent::LoadModel(UNNEModelData* ModelData)
{
    if (!ModelData)
    {
        UE_LOG(LogTemp, Error, TEXT("NNE: ModelData is null"));
        return false;
    }

    // 既存モデルを解放
    ReleaseModel();

    // GPU Runtime取得（動作確認済み）
    TWeakInterfacePtr<INNERuntimeGPU> Runtime =
        UE::NNE::GetRuntime<INNERuntimeGPU>(FString("NNERuntimeORTDml"));

    if (!Runtime.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("NNE: GPU Runtime not found"));
        return false;
    }

    // モデル作成
    TSharedPtr<UE::NNE::IModelGPU> Model = Runtime->CreateModelGPU(ModelData);
    if (!Model.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("NNE: Failed to create GPU model"));
        return false;
    }

    ModelInstance = Model->CreateModelInstanceGPU();
    if (!ModelInstance.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("NNE: Failed to create model instance"));
        return false;
    }

    // 入力形状を設定（batch=1, features=3）
    TArray<UE::NNE::FTensorShape> InputShapes;
    InputShapes.Add(UE::NNE::FTensorShape::Make({ 1, 3 }));
    ModelInstance->SetInputTensorShapes(InputShapes);

    UE_LOG(LogTemp, Log, TEXT("NNE: Model loaded successfully"));
    return true;
}

TArray<float> UNNEManagerComponent::RunInference(const TArray<float>& InputData)
{
    TArray<float> OutputData = { 0.0f, 0.0f, 0.0f };

    if (!ModelInstance.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("NNE: No model loaded"));
        return OutputData;
    }

    TArray<UE::NNE::FTensorBindingCPU> Inputs, Outputs;
    Inputs.Add({ (void*)InputData.GetData(), (uint64)InputData.Num() * sizeof(float) });
    Outputs.Add({ OutputData.GetData(), (uint64)OutputData.Num() * sizeof(float) });

    ModelInstance->RunSync(Inputs, Outputs);

    return OutputData;
}

void UNNEManagerComponent::ReleaseModel()
{
    ModelInstance.Reset();
}