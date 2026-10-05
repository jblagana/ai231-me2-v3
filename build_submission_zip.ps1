# =============================================================
#  ME2 V3 - build the Google Classroom submission zip
#
#  Packages the "released separately" artifacts (final model +
#  training logs) + provenance docs into ONE zip for Classroom.
#
#  Run AFTER pulling the HPC training logs into runs\v3r1\
#  (see runs\v3r1\README.md). If the logs are still missing it
#  warns and still builds (model + docs only).
# =============================================================
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$stage = Join-Path $PSScriptRoot '_release_staging'
$zip   = Join-Path $PSScriptRoot 'ai231-me2-v3-submission.zip'
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Path (Join-Path $stage 'runs') -Force | Out-Null
$runOut = Join-Path $stage 'runs\v3r1'
New-Item -ItemType Directory -Path $runOut -Force | Out-Null

# 1) provenance + submission docs (all the links live here)
Copy-Item 'RELEASE_README.md' $stage
Copy-Item 'SUBMISSION.md'     $stage

# 2) final model
Copy-Item '_pi_app\me2_vcm_v3.onnx' (Join-Path $stage 'me2_vcm_v3.onnx')

# 3) training logs (whitelist the small artifacts; skip big *.npz caches)
$runSrc = Join-Path $PSScriptRoot 'runs\v3r1'
$keep   = @('results.json','train.log','launch.out','per_class_test.csv','slot_confusion_test.csv','checkpoint_best.pt','README.md')
$have   = @()
if (Test-Path $runSrc) {
  foreach ($f in $keep) {
    $s = Join-Path $runSrc $f
    if (Test-Path $s) { Copy-Item $s $runOut; $have += $f }
  }
}
if ($have.Count -eq 0) {
  Write-Warning 'No training logs found in runs\v3r1 - the HPC pull has not happened yet.'
  Write-Warning 'The zip will contain the model + docs but NOT the training logs.'
} else {
  Write-Host ('Packaged training logs: ' + ($have -join ', '))
}

# 4) zip it up
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip -Force
Remove-Item $stage -Recurse -Force

Write-Host ''
Write-Host 'Created:' -ForegroundColor Green
Get-Item $zip | Format-List Name, @{n='KB';e={[math]::Round($_.Length/1KB,1)}}, FullName
Write-Host 'Upload this zip to Google Classroom, and paste the repo + dataset links in the submission text.'
