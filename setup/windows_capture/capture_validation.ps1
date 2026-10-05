# Pixel-only validation. This helper does not query or operate the desktop.
Add-Type -AssemblyName System.Drawing
if (-not ('WslShotPixelAuditV1' -as [type])) {
    $drawingReferences = @([System.Drawing.Bitmap].Assembly.Location, [System.Drawing.Color].Assembly.Location) | Select-Object -Unique
    Add-Type -ErrorAction Stop -ReferencedAssemblies $drawingReferences -TypeDefinition @"
using System;
using System.Drawing;
public sealed class WslShotPixelStatsV1 {
    public int Width, Height, SampleStride, SampleCount;
    public double NearBlackFraction, MeanLuminance, MaximumLuminance;
    public bool Rejected;
}
public static class WslShotPixelAuditV1 {
    public static WslShotPixelStatsV1 Measure(Bitmap bitmap) {
        if (bitmap == null || bitmap.Width < 1 || bitmap.Height < 1)
            throw new ArgumentException("Empty capture bitmap");
        // Uniform grid of at least ~65k samples at 4K; no bitmap mutation.
        int stride = Math.Max(1, (int)Math.Sqrt((double)bitmap.Width * bitmap.Height / 65536.0));
        int count = 0, dark = 0;
        double total = 0, maximum = 0;
        for (int y = 0; y < bitmap.Height; y += stride) {
            for (int x = 0; x < bitmap.Width; x += stride) {
                Color c = bitmap.GetPixel(x, y);
                double luminance = 0.2126 * c.R + 0.7152 * c.G + 0.0722 * c.B;
                if (c.R <= 12 && c.G <= 12 && c.B <= 12) dark++;
                total += luminance;
                maximum = Math.Max(maximum, luminance);
                count++;
            }
        }
        double fraction = (double)dark / count, mean = total / count;
        return new WslShotPixelStatsV1 {
            Width = bitmap.Width, Height = bitmap.Height, SampleStride = stride,
            SampleCount = count, NearBlackFraction = fraction, MeanLuminance = mean,
            MaximumLuminance = maximum,
            Rejected = fraction >= 0.98 && mean <= 12.0
        };
    }
}
"@
}

function Get-WslShotImageQuality {
    param([Parameter(Mandatory=$true)][System.Drawing.Bitmap]$Bitmap)
    $stats = [WslShotPixelAuditV1]::Measure($Bitmap)
    return [pscustomobject][ordered]@{
        validator = 'near_black_grid_v1'
        width = $stats.Width
        height = $stats.Height
        sample_stride = $stats.SampleStride
        sampled_pixels = $stats.SampleCount
        near_black_rgb_max = 12
        rejection_fraction_threshold = 0.98
        near_black_fraction = $stats.NearBlackFraction
        mean_luminance = $stats.MeanLuminance
        maximum_luminance = $stats.MaximumLuminance
        accepted_not_black = (-not $stats.Rejected)
        limitation = 'Pixel visibility gate only; human review must check terminal content, theme, popups and command identity.'
    }
}

function Assert-WslShotImageFile {
    param([Parameter(Mandatory=$true)][string]$Path)
    $image = [System.Drawing.Bitmap]::new($Path)
    try { $quality = Get-WslShotImageQuality -Bitmap $image }
    finally { $image.Dispose() }
    if (-not $quality.accepted_not_black) {
        throw "BLACK_CAPTURE_REJECTED: $Path; near-black fraction=$($quality.near_black_fraction), mean luminance=$($quality.mean_luminance). Check display/session/lock state; do not rerun the experiment automatically."
    }
    return $quality
}
