param(
    [string]$MqttHost = "127.0.0.1",
    [int]$MqttPort = 1883,
    [switch]$NoUI,
    [switch]$Simulate
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$launchArgs = @("launcher.py", "--mqtt-host", $MqttHost, "--mqtt-port", "$MqttPort")
if ($NoUI) { $launchArgs += "--no-ui" }
if ($Simulate) { $launchArgs += "--simulate" }

Push-Location $root
try {
    & python @launchArgs
} finally {
    Pop-Location
}
