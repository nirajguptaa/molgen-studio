$ErrorActionPreference = 'Stop'

$targetDir = Join-Path $PSScriptRoot '..\tools\vina'
$targetDir = [System.IO.Path]::GetFullPath($targetDir)
New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

$vinaPath = Join-Path $targetDir 'vina_1.2.7_win.exe'
$releaseUrl = 'https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_win.exe'

if (-not (Test-Path -LiteralPath $vinaPath)) {
    & curl.exe --fail --location --silent --show-error $releaseUrl --output $vinaPath
    if ($LASTEXITCODE -ne 0) {
        throw "Could not download AutoDock Vina from the official release URL: $releaseUrl"
    }
}

$version = & $vinaPath --version
if ($LASTEXITCODE -ne 0 -or $version -notmatch 'AutoDock Vina v1\.2\.7') {
    throw "Unexpected Vina executable/version at $vinaPath. Expected AutoDock Vina 1.2.7."
}

Write-Output "Ready: $version"
