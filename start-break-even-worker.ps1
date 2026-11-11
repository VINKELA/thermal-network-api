$ErrorActionPreference = 'Stop'
$existing = Get-CimInstance Win32_Process -Filter "name='python.exe'" |
    Where-Object { $_.CommandLine -match 'celery' -and $_.CommandLine -match 'break-even@' -and $_.CommandLine -match 'break_even' }
if ($existing) {
    Write-Output 'The dedicated break-even Celery worker is already running.'
    return
}
$logRoot = Join-Path $PSScriptRoot 'logs'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$worker = Start-Process -FilePath (Join-Path $PSScriptRoot 'venv\Scripts\python.exe') `
    -ArgumentList @('-m', 'celery', '-A', 'network', 'worker', '--pool=solo', '--concurrency=1',
                    '--queues=break_even', '--hostname=break-even@%h', '--loglevel=INFO') `
    -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logRoot 'break-even-worker.out.log') `
    -RedirectStandardError (Join-Path $logRoot 'break-even-worker.err.log')
Write-Output "Started the break-even Celery worker (process $($worker.Id))."
