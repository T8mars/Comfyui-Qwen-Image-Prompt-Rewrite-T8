$ErrorActionPreference = 'Stop'
$base = Join-Path (Split-Path -Parent $PSScriptRoot) 'runtime\llama-b11068'
New-Item -ItemType Directory -Force -Path $base | Out-Null
$artifacts = @(
    @{ Name = 'llama-b11068-bin-win-cuda-12.4-x64.zip'; Sha256 = '4464f3bc733dacdf62f4d66a79703bab8ab184235905bcf63a3643b802530196' },
    @{ Name = 'cudart-llama-bin-win-cuda-12.4-x64.zip'; Sha256 = '8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6' }
)
foreach ($artifact in $artifacts) {
    $name = $artifact.Name
    $zip = Join-Path $base $name
    if (!(Test-Path -LiteralPath $zip)) {
        $url = 'https://github.com/ggml-org/llama.cpp/releases/download/b11068/' + $name
        $partial = "$zip.download"
        Invoke-WebRequest -Uri $url -OutFile $partial -TimeoutSec 600
        if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $artifact.Sha256) {
            throw "Downloaded archive failed SHA256 check: $name. Delete $partial and retry."
        }
        Move-Item -LiteralPath $partial -Destination $zip
    }
    if ((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash -ne $artifact.Sha256) {
        throw "Archive failed SHA256 check: $zip. Delete it and rerun this script."
    }
    Expand-Archive -LiteralPath $zip -DestinationPath $base -Force
    Write-Output "extracted $name"
}
$server = Join-Path $base 'llama-server.exe'
if (!(Test-Path -LiteralPath $server)) {
    throw "llama-server.exe was not found after extraction in $base. See README.md, Install llama-server (Windows)."
}
Get-Item -LiteralPath $server | Select-Object FullName,Length
