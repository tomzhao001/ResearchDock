Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$EnvFile = if ($env:ENV_FILE) { $env:ENV_FILE } else { Join-Path $RepoRoot ".env" }

function Import-EnvFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    foreach ($rawLine in Get-Content -LiteralPath $Path) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            continue
        }

        $parts = $line -split "=", 2
        if ($parts.Count -ne 2) {
            continue
        }

        $name = $parts[0].Trim()
        $value = $parts[1]

        if ([string]::IsNullOrWhiteSpace($name)) {
            continue
        }

        if (-not (Test-Path "Env:$name")) {
            if (
                ($value.Length -ge 2) -and
                (
                    ($value.StartsWith('"') -and $value.EndsWith('"')) -or
                    ($value.StartsWith("'") -and $value.EndsWith("'"))
                )
            ) {
                $value = $value.Substring(1, $value.Length - 2)
            }

            Set-Item -Path "Env:$name" -Value $value
        }
    }
}

function Require-Env {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) {
        throw "Missing required environment variable: $Name"
    }

    return $value
}

Import-EnvFile -Path $EnvFile

$acrRegistry = Require-Env "ACR_REGISTRY"
$imageNamespace = Require-Env "IMAGE_NAMESPACE"

$imageTag = if ($env:IMAGE_TAG) { $env:IMAGE_TAG } else { "latest" }

$backendImage = "$acrRegistry/$imageNamespace/backend`:$imageTag"
$frontendImage = "$acrRegistry/$imageNamespace/frontend`:$imageTag"

Write-Host "Building $backendImage"
docker build -t $backendImage (Join-Path $RepoRoot "backend")

Write-Host "Building $frontendImage"
docker build -t $frontendImage (Join-Path $RepoRoot "frontend")

Write-Host "Pushing $backendImage"
docker push $backendImage

Write-Host "Pushing $frontendImage"
docker push $frontendImage

Write-Host "Done."
