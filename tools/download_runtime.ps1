$ErrorActionPreference = 'Stop'
$base = Join-Path (Split-Path -Parent $PSScriptRoot) 'runtime\llama-b11068'
New-Item -ItemType Directory -Force -Path $base | Out-Null
$artifacts = @(
    'llama-b11068-bin-win-cuda-12.4-x64.zip',
    'cudart-llama-bin-win-cuda-12.4-x64.zip'
)
foreach ($name in $artifacts) {
    $zip = Join-Path $base $name
    if (!(Test-Path -LiteralPath $zip)) {
        $url = 'https://github.com/ggml-org/llama.cpp/releases/download/b11068/' + $name
        Invoke-WebRequest -Uri $url -OutFile $zip -TimeoutSec 600
    }
    Expand-Archive -LiteralPath $zip -DestinationPath $base -Force
    Write-Output "extracted $name"
}
Get-ChildItem -LiteralPath $base -Recurse -Filter llama-server.exe | Select-Object FullName,Length
