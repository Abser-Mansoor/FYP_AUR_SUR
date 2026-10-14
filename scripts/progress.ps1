param([string]$Log = "D:\FYP_AUR_SUR\outputs\results\phase1_resume.log", [string]$Ref = "D:\FYP_AUR_SUR\outputs\results\phase1\run.log", [int]$Loop = 0)

function Show-Progress {
    $recs = "arcface", "adaface", "magface", "curricularface"
    $lines = @(Get-Content $Log | ForEach-Object { $_ -replace "`0", "" } | Where-Object { $_ -match '^\[\d{4}-' })
    $ts = { param($l) [datetime]::ParseExact($l.Substring(1, 23), "yyyy-MM-dd HH:mm:ss,fff", $null) }

    $starts = @(0..($lines.Count - 1) | Where-Object { $lines[$_] -match '\] (\w+): weights sha256' })
    if (-not $starts) { Write-Host "No recognizer started yet."; return }
    $blocks = for ($i = 0; $i -lt $starts.Count; $i++) {
        $end = if ($i + 1 -lt $starts.Count) { $starts[$i + 1] - 1 } else { $lines.Count - 1 }
        [pscustomobject]@{ Name = ([regex]::Match($lines[$starts[$i]], '\] (\w+): weights')).Groups[1].Value
                           Lines = @($lines[$starts[$i]..$end] | Where-Object { $_ -notmatch '^\[[^\]]+\] -{10}' }) }
    }
    $cur = $blocks[-1]
    $finished = ($cur.Lines -match 'ALIGNED \(diagnostic\)|baseline complete').Count -gt 0
    $tmpl = $blocks | Where-Object { $_ -ne $cur -and ($_.Lines -match 'ALIGNED \(diagnostic\)').Count } | Select-Object -Last 1

    $n = $cur.Lines.Count
    $last = $cur.Lines[-1]
    $elapsed = ((Get-Date) - (& $ts $last)).TotalSeconds
    $clean = { param($l) ($l -replace '^\[[^\]]+\]\s*', '') -replace '(\{.*)', '{...}' }

    Write-Host ""
    Write-Host ("Recognizers done: " + (($blocks | Where-Object { $_ -ne $cur -or $finished }).Name -join ", ")) -ForegroundColor DarkGray
    $idx = [array]::IndexOf($recs, $cur.Name) + 1
    Write-Host "CURRENT RECOGNIZER : $($cur.Name)  ($idx of $($recs.Count))$(if ($finished) {' - finished'})" -ForegroundColor Cyan
    Write-Host "LAST COMPLETED STEP: $(& $clean $last)  ($([int]$elapsed)s ago)"

    $key = { param($l) (($l -replace '^\[[^\]]+\]\s*', '') -replace '\d[\d.]*s?\b.*$', '') -replace '^\w+(?= sanity| ALIGNED)', 'REC' }
    $refLines = @(Get-Content $Ref | ForEach-Object { $_ -replace "`0", "" } | Where-Object { $_ -match '^\[\d{4}-' }) + $lines | Where-Object { $_ -notmatch '\] [-=]{10}' }
    $k = & $key $last
    $found = $null
    for ($i = $refLines.Count - 2; $i -ge 0; $i--) {
        if ((& $key $refLines[$i]) -eq $k -and $refLines[$i + 1] -notmatch 'weights sha256|Phase 1 baseline|^\[[^\]]+\] preprocess') { $found = $i; break }
    }
    if ($found -ne $null -and -not $finished) {
        $nextLine = $refLines[$found + 1]
        $dur = ((& $ts $nextLine) - (& $ts $refLines[$found])).TotalSeconds
        $pct = [math]::Min(99, [math]::Round(100 * $elapsed / [math]::Max($dur, 1)))
        Write-Host "RUNNING STAGE      : $(& $clean $nextLine)" -ForegroundColor Yellow
        Write-Host "STAGE PROGRESS     : ~$pct%  ($([int]$elapsed)s of ~$([int]$dur)s typical; can exceed on a slower stage)" -ForegroundColor Green
        Write-Host "STEPS DONE (this recognizer): $n" -ForegroundColor Green
    } elseif (-not $finished) {
        Write-Host "RUNNING STAGE      : (unknown - no reference for this step)" -ForegroundColor Yellow
    }
    Write-Host "GPU                : $((nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) -join '')"
}

if ($Loop -gt 0) { while ($true) { Clear-Host; Show-Progress; Start-Sleep -Seconds $Loop } } else { Show-Progress }




