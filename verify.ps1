#
# verify.ps1 - End-to-end sanity check after editing source files.
#
# Usage (from project root):
#     powershell -ExecutionPolicy Bypass -File .\verify.ps1
#
# What it does:
#   1. Verifies Python imports still work (catches syntax errors).
#   2. Re-trains the model from mtcars.csv (catches data/script regressions).
#   3. Runs the pytest suite (10 tests).
#   4. Boots the API locally with uvicorn and hits /health, /ready, /predict.
#   5. Hits the deployed Cloud Run service on the same three endpoints.
#
# Exits with a clear PASS/FAIL summary at the end.
#

$ErrorActionPreference = "Stop"

# ---- configuration --------------------------------------------------------
$ProjectRoot   = "C:\Users\Administrator\Desktop\assignment7"
$LocalPort     = 8090   # use a non-default port to avoid clashing with other servers
$LocalBase     = "http://127.0.0.1:$LocalPort"
$CloudRunBase  = "https://mtcars-fastapi-509303146040.us-central1.run.app"
$PredictBody   = '{"wt": 2.62, "hp": 110}'

# Track results so we can print a summary
$Results = @()
function Record-Step($name, $ok, $details = "") {
    $script:Results += [pscustomobject]@{ Step = $name; OK = $ok; Details = $details }
    if ($ok) {
        Write-Host "  [PASS] $name $details" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] $name $details" -ForegroundColor Red
    }
}

function Section($title) {
    Write-Host ""
    Write-Host "==== $title ====" -ForegroundColor Cyan
}

Set-Location $ProjectRoot

# ---- 1. python import smoke test -----------------------------------------
Section "1. Python import smoke test"
try {
    $out = & python -c "from app.main import app; from scripts import train_model; print('IMPORT_OK')" 2>&1
    if ($out -match "IMPORT_OK") {
        Record-Step "imports" $true
    } else {
        Record-Step "imports" $false ($out -join "; ")
    }
} catch {
    Record-Step "imports" $false $_.Exception.Message
}

# ---- 2. retrain the model ------------------------------------------------
Section "2. Retrain model"
try {
    $out = & python scripts\train_model.py 2>&1
    if ($LASTEXITCODE -eq 0 -and (Test-Path "models\model.pkl")) {
        # Pull the R^2 line out of the output so the user sees it
        $r2line = ($out | Select-String "R\^2  =").Line
        Record-Step "retrain" $true $r2line
    } else {
        Record-Step "retrain" $false ($out -join "; ")
    }
} catch {
    Record-Step "retrain" $false $_.Exception.Message
}

# ---- 3. pytest -----------------------------------------------------------
Section "3. pytest"
try {
    $out = & python -m pytest -q 2>&1
    if ($LASTEXITCODE -eq 0) {
        $summary = ($out | Select-String "passed").Line
        Record-Step "pytest" $true $summary
    } else {
        Record-Step "pytest" $false ($out -join "; ")
    }
} catch {
    Record-Step "pytest" $false $_.Exception.Message
}

# ---- 4. local uvicorn smoke test -----------------------------------------
Section "4. Local API smoke test (uvicorn on port $LocalPort)"

# Boot uvicorn in the background
$uvicornArgs = "-m uvicorn app.main:app --host 127.0.0.1 --port $LocalPort"
$proc = Start-Process -FilePath "python" -ArgumentList $uvicornArgs `
    -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput "$env:TEMP\uvi_stdout.log" `
    -RedirectStandardError  "$env:TEMP\uvi_stderr.log"

# Wait up to 15 s for the server to start
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Milliseconds 500
    try {
        $r = Invoke-WebRequest -Uri "$LocalBase/health" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch {}
}

if (-not $ready) {
    Record-Step "local-uvicorn-boot" $false "server failed to start within 15s"
} else {
    Record-Step "local-uvicorn-boot" $true

    # /health
    try {
        $r = Invoke-RestMethod "$LocalBase/health"
        Record-Step "local /health" ($r.status -eq "ok") "status=$($r.status)"
    } catch {
        Record-Step "local /health" $false $_.Exception.Message
    }

    # /ready
    try {
        $r = Invoke-RestMethod "$LocalBase/ready"
        Record-Step "local /ready" ($r.model_loaded -eq $true) "model_loaded=$($r.model_loaded)"
    } catch {
        Record-Step "local /ready" $false $_.Exception.Message
    }

    # /predict (valid)
    try {
        $r = Invoke-RestMethod -Uri "$LocalBase/predict" -Method POST -ContentType "application/json" -Body $PredictBody
        $mpg = [double]$r.predicted_mpg
        $ok = ($mpg -gt 20 -and $mpg -lt 27)
        Record-Step "local /predict" $ok "predicted_mpg=$mpg"
    } catch {
        Record-Step "local /predict" $false $_.Exception.Message
    }

    # /predict missing field -> expect 422 (works on both PS 5 and PS 7).
    # PS 5 throws on 4xx, so we read the status code out of the exception.
    $statusCode = $null
    try {
        $r = Invoke-WebRequest -Uri "$LocalBase/predict" -Method POST -ContentType "application/json" `
            -Body '{"wt":2.62}' -UseBasicParsing
        $statusCode = $r.StatusCode
    } catch {
        if ($_.Exception.Response) {
            $statusCode = [int]$_.Exception.Response.StatusCode
        }
    }
    Record-Step "local /predict missing-field" ($statusCode -eq 422) "status=$statusCode"
}

# Stop uvicorn
try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch {}

# ---- 5. Cloud Run public smoke test --------------------------------------
Section "5. Cloud Run public API"

# /health
try {
    $r = Invoke-RestMethod "$CloudRunBase/health" -TimeoutSec 30
    Record-Step "cloudrun /health" ($r.status -eq "ok") "status=$($r.status)"
} catch {
    Record-Step "cloudrun /health" $false $_.Exception.Message
}

# /ready
try {
    $r = Invoke-RestMethod "$CloudRunBase/ready" -TimeoutSec 30
    Record-Step "cloudrun /ready" ($r.model_loaded -eq $true) "model_loaded=$($r.model_loaded)"
} catch {
    Record-Step "cloudrun /ready" $false $_.Exception.Message
}

# /predict
try {
    $r = Invoke-RestMethod -Uri "$CloudRunBase/predict" -Method POST `
        -ContentType "application/json" -Body $PredictBody -TimeoutSec 30
    $mpg = [double]$r.predicted_mpg
    $ok = ($mpg -gt 20 -and $mpg -lt 27)
    Record-Step "cloudrun /predict" $ok "predicted_mpg=$mpg"
} catch {
    Record-Step "cloudrun /predict" $false $_.Exception.Message
}

# ---- summary -------------------------------------------------------------
Write-Host ""
Write-Host "==== Summary ====" -ForegroundColor Cyan
$passed = ($Results | Where-Object { $_.OK }).Count
$failed = ($Results | Where-Object { -not $_.OK }).Count
foreach ($r in $Results) {
    $color = if ($r.OK) { "Green" } else { "Red" }
    $mark  = if ($r.OK) { "PASS" } else { "FAIL" }
    Write-Host ("  [{0}] {1,-40} {2}" -f $mark, $r.Step, $r.Details) -ForegroundColor $color
}
Write-Host ""
Write-Host "Total: $passed passed, $failed failed" -ForegroundColor $(if ($failed -eq 0) { "Green" } else { "Red" })
if ($failed -gt 0) { exit 1 }
