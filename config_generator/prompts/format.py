from pydantic import BaseModel


class TestParameters(BaseModel):
    marks: list[str]


class GlobalParameters(BaseModel):
    MinimumRequiredVersion: str = "5.0.0"
    SleepPercent: int = 50
    NumElementsToValidate: int = 128
    DataInitTypeBeta: int = 0
    DataInitTypeAlpha: int = 1
    CSVExportWinner: int = 1
    CSVMergeSameProblemID: int = 1
    Device: int = 0
    NumWarmups: int = 1000
    EnqueuesPerSync: int = 100
    PrintSolutionRejectionReason: bool = True
    PrintLevel: int = (
        2  # 0 - user wants no printing, 1 - user wants limited prints (for tuning), 2 - user wants full prints
    )
    ClientLogLevel: int = (
        3  # Error = 0 (crash), Terse = 1(no predicator info, for tuning), Verbose = 2, Debug = 3(for deubg to see predicator)
    )
    KeepBuildTmp: bool = True


class ProblemType(BaseModel):
    OperationType: str = "GEMM"
    DataType: str = "h"  # fp16
    DestDataType: str = "h"  # fp16
    ComputeDataType: str = "s"  # fp32
    HighPrecisionAccumulate: bool = True
    TransposeA: int = 0
    TransposeB: int = 0
    UseBeta: bool = True
    UseBias: int = 1
    Batched: bool = True
    Activation: bool = False
    ActivationType: str | None = None


class BenchmarkCommonParameter(BaseModel):
    KernelLanguage: list[str] = ["Assembly"]


class MatrixInstruction(BaseModel):
    MatrixInstruction: list[list[int]]


class ExactSize(BaseModel):
    Exact: list[int]


class ProblemSizes(BaseModel):
    ProblemSizes: list[ExactSize]


class BiasTypeArgs(BaseModel):
    BiasTypeArgs: list[str] = ["h"]


class BenchmarkProblemSizeGroup(BaseModel):
    InitialSolutionParameters: list | None = None
    BenchmarkCommonParameters: list[BenchmarkCommonParameter]
    ForkParameters: list[MatrixInstruction] | None = None
    BenchmarkJoinParameters: list | None = None
    BenchmarkFinalParameters: list[ProblemSizes | BiasTypeArgs]


class ConfigYaml(BaseModel):
    TestParameters: TestParameters
    GlobalParameters: GlobalParameters
    BenchmarkProblems: list[list[ProblemType | BenchmarkProblemSizeGroup]]


if __name__ == "__main__":
    import sys
    import yaml

    with open(sys.argv[1], "r") as f:
        config: dict = yaml.safe_load(f)

    y = ConfigYaml.model_validate(config)  # recommended way in pydantic v2
    # y = ConfigYaml(**config)

    print(y)
    print(type(y))
