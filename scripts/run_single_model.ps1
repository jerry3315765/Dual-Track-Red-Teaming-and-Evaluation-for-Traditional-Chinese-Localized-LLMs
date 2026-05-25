param(
  [Parameter(Mandatory=$true)][string]$ModelName
)

python scripts/run_single_model.py --model_name $ModelName
