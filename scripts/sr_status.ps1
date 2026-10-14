param([int]$Loop = 0)
# Phase 3 status: what is done, what remains, estimated finish. Reads files only (no GPU calls).
#   powershell -NoProfile -File scripts\sr_status.ps1            (once)
#   powershell -NoProfile -File scripts\sr_status.ps1 -Loop 60   (refresh every 60 s)
$root = "D:\FYP_AUR_SUR"
$nTune = 2800; $nEval = 22898
# isolated seconds per crop (smoke-test medians); settings are the tuning sweep
$sweep = @(
  @('bicubic','pre128',0.005), @('bicubic','pre512',0.005),
  @('esrgan','native',0.073), @('esrgan','pre64',0.059), @('esrgan','pre128',0.059),
  @('invsr','pre64_st1',0.122), @('invsr','pre128_st1',0.243), @('invsr','pre256_st1',1.143), @('invsr','pre128_st3',0.33), @('invsr','pre128_st5',0.423),
  @('tadsr','pre64_t500',0.256), @('tadsr','pre128_t500',0.47), @('tadsr','pre256_t500',1.627), @('tadsr','pre128_t250',0.47), @('tadsr','pre128_t750',0.47),
  @('osdface','pre128',0.50), @('osdface','pre512',0.50)
)
$assumed = @{ bicubic = @('pre512',0.005); esrgan = @('native',0.073); invsr = @('pre128_st1',0.243); tadsr = @('pre128_t500',0.47); osdface = @('pre128',0.50) }
$stochastic = @('invsr','tadsr','osdface')

function Count-Png($dir) { if (Test-Path $dir) { ([System.IO.Directory]::EnumerateFiles($dir, '*.png') | Measure-Object).Count } else { 0 } }
function Show-Status {
  $rem = 0.0; $rows = @()
  foreach ($v in $sweep) {
    $arm = $v[0]; $set = $v[1]; $sp = $v[2]
    $d = Count-Png "$root\outputs\sr\restored\tuning\${arm}__${set}\seed0"
    $left = [math]::Max($nTune - $d, 0); $rem += $left * $sp
    $rows += [pscustomobject]@{ Stage = 'tuning'; Variant = "${arm}__${set}"; Done = $d; Total = $nTune; Pct = [math]::Round(100 * $d / $nTune); RemMin = [math]::Round($left * $sp / 60) }
    if (-not (Test-Path "$root\outputs\sr\results\tuning_${arm}.csv") -and $left -eq 0) { }
  }
  foreach ($arm in 'bicubic','esrgan','invsr','tadsr','osdface') {
    $selFile = "$root\outputs\sr\results\selection_${arm}.json"
    $set = $assumed[$arm][0]; $sp = $assumed[$arm][1]; $tag = ' (assumed best)'
    if (Test-Path $selFile) { $j = Get-Content $selFile -Raw | ConvertFrom-Json; if ($j) { $set = @($j)[0].setting; $tag = ''; $m = $sweep | Where-Object { $_[0] -eq $arm -and $_[1] -eq $set }; if ($m) { $sp = $m[2] } } }
    $d = Count-Png "$root\outputs\sr\restored\eval\${arm}__${set}\seed0"
    $left = [math]::Max($nEval - $d, 0); $rem += $left * $sp
    $rows += [pscustomobject]@{ Stage = 'eval'; Variant = "${arm}__${set}$tag"; Done = $d; Total = $nEval; Pct = [math]::Round(100 * $d / $nEval); RemMin = [math]::Round($left * $sp / 60) }
    if ($stochastic -contains $arm) {
      foreach ($sd in 1,2) {
        $d = Count-Png "$root\outputs\sr\restored\tuning\${arm}__${set}\seed$sd"
        $left = [math]::Max($nTune - $d, 0); $rem += $left * $sp
        $rows += [pscustomobject]@{ Stage = "seeds"; Variant = "${arm}__${set}$tag seed$sd"; Done = $d; Total = $nTune; Pct = [math]::Round(100 * $d / $nTune); RemMin = [math]::Round($left * $sp / 60) }
      }
    }
    # scoring overhead still pending (recognizer embeddings, bootstrap CIs)
    if (-not (Test-Path "$root\outputs\sr\results\eval_${arm}.csv")) { $rem += 420 }
    if (($stochastic -contains $arm) -and -not (Test-Path "$root\outputs\sr\results\seed_study_${arm}.csv")) { $rem += 300 }
    if (-not (Test-Path "$root\outputs\sr\results\tuning_${arm}.csv")) { $rem += 90 * @($sweep | Where-Object { $_[0] -eq $arm }).Count }
  }
  $procs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'sr_pipeline.py' }).Count
  $now = Get-Date
  Write-Host ""
  Write-Host ("Phase 3 status at {0}" -f $now.ToString('yyyy-MM-dd HH:mm:ss')) -ForegroundColor Cyan
  Write-Host ("Worker running: {0}  (2 processes = 1 worker)  {1}" -f $(if ($procs -ge 1) { 'YES' } else { 'NO' }), $(if ($procs -ge 1) { '' } else { '-> restart with:  schtasks /run /tn SR_worker  (resumes where it stopped)' }))
  $rows | Where-Object { $_.Done -lt $_.Total } | Format-Table Stage, Variant, Done, Total, Pct, RemMin -AutoSize
  $doneV = @($rows | Where-Object { $_.Done -ge $_.Total }).Count
  Write-Host ("Finished stages: {0} of {1}" -f $doneV, $rows.Count)
  $hrs = $rem / 3600
  Write-Host ("Remaining GPU work (isolated-speed estimate): {0:N1} h  ->  estimated finish about {1}  (+-25%; one worker, jobs run one after another)" -f $hrs, $now.AddSeconds($rem).ToString('yyyy-MM-dd HH:mm')) -ForegroundColor Green
  foreach ($f in Get-ChildItem "$root\outputs\sr\results" -Filter 'needs_approval_*.txt' -ErrorAction SilentlyContinue) { Write-Host ("ATTENTION: " + (Get-Content $f.FullName -Raw)) -ForegroundColor Yellow }
  Write-Host "Last log lines:"; Get-Content "$root\outputs\sr\run.log" -Tail 4 | ForEach-Object { '  ' + $_.Substring(0, [math]::Min(150, $_.Length)) }
}
if ($Loop -gt 0) { while ($true) { Clear-Host; Show-Status; Start-Sleep -Seconds $Loop } } else { Show-Status }
