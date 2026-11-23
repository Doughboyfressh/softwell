/**
 * bench/targets.ts — the enterprise hardware targets the virtual
 * machine must beat, with sources. Verified 2026-11-22.
 *
 * Which chip is "the best" and why:
 * - vGPU: the genuinely current best enterprise GPU is the NVIDIA
 *   Rubin R100 (in full production since early 2026, shipping H2
 *   2026; NVIDIA HGX page). It is the PRIMARY target. The Blackwell
 *   B200 is kept as a SECONDARY target because it is the widely
 *   deployed reference and the number the user originally named.
 *   Rubin per GPU (NVIDIA HGX Rubin NVL8 page): 35 PFLOPS dense FP4,
 *   17.5 PFLOPS FP8, 4 PFLOPS FP16/BF16, 288 GB HBM4, 130 TB/s for
 *   8 GPUs = 16.25 TB/s per GPU. Other sources quote up to 19.2–22
 *   TB/s per GPU; we use the conservative NVIDIA system figure
 *   (16.25 TB/s) so the target is the hardest defensible one.
 *   B200 per GPU (NVIDIA datasheet): 9,000 TFLOPS dense FP4, 4,500
 *   FP8, 2,250 BF16, 8 TB/s HBM3e, 192 GB.
 * - vCPU: AMD EPYC 9965 "Turin" (Oct 2024) — still the shipping
 *   core-count leader; EPYC 9006 "Venice" has been announced but
 *   no shipping specs to benchmark against. 192 cores / 384
 *   threads, 384 MB L3, 12-channel DDR5: 576 GB/s at DDR5-6000
 *   (TechPowerUp) and up to 614 GB/s at DDR5-6400 (AMD). We use
 *   614 GB/s (AMD's up-to figure). FP32 peak is an ESTIMATE from
 *   first principles, not a vendor number: 192 cores x 2.25 GHz
 *   base x 32 FP32 FLOPs/cycle (2x AVX-512 FMA) = 27.6 TFLOPS;
 *   labelled as such wherever it appears.
 * - vRAM: HBM3e as shipped on the B200 — 8 TB/s, 192 GB per device.
 *   (Rubin's HBM4 is 16.25 TB/s / 288 GB; vRAM is compared against
 *   HBM3e because the user's target names HBM3e, and the Rubin
 *   memory numbers appear in the vGPU bandwidth line instead.)
 * - vSSD: Micron 9650 PCIe Gen6 (announced Jul 2025, in volume
 *   production) — 28 GB/s sequential read, 14 GB/s sequential
 *   write, 5.5M random read IOPS, 900K random write IOPS.
 *   Still the fastest shipping data-center SSD (Micron product
 *   brief; Tom's Hardware, Jul 2025).
 * - vNPU: Google TPU v6e (Trillium) is the PRIMARY target (1,836
 *   INT8 TOPS, 918 BF16 TFLOPS, 1,638 GB/s, 32 GB); TPU v5p is the
 *   secondary (918 INT8 TOPS, 459 BF16/FP8 TFLOPS, 2,765 GB/s,
 *   95 GiB). Trainium2 and Inferentia2 are secondaries too — see
 *   VNPU_TARGETS and docs/VNPU.md §5.2 for the full table and the
 *   v5p INT8-vs-FP8 provenance note.
 *
 * Sources:
 * - https://www.nvidia.com/en-us/data-center/hgx/
 * - https://www.spheron.network/blog/nvidia-b200-complete-guide/
 * - https://www.techpowerup.com/cpu-specs/epyc-9965.c3904
 * - https://www.cpu-monkey.com/en/cpu-amd_epyc_9965
 * - https://my.micron.com/products/storage/ssd/data-center-ssd/9650-ssd
 * - https://www.tomshardware.com/pc-components/ssds/microns-industry-first-pci-6-0-ssd-promises-sequential-reads-up-to-28-000-mb-s-245-tb-ssd-also-coming-for-those-who-need-capacity-more-than-cutting-edge-speed
 */

export interface DeviceTargets {
  /** Primary target chip name. */
  primary: string;
  /** Secondary (older / reference) target chip name, if any. */
  secondary?: string;
  /** Primary-target numbers. */
  primaryMetrics: Record<string, number>;
  /** Secondary-target numbers (same keys as primaryMetrics). */
  secondaryMetrics?: Record<string, number>;
  /** Where the numbers came from. */
  sources: string[];
}

export const VGPU_TARGETS: DeviceTargets = {
  primary: "NVIDIA Rubin R100",
  secondary: "NVIDIA B200 (Blackwell)",
  primaryMetrics: {
    fp4Tflops: 35_000, // dense FP4, PFLOPS -> TFLOPS
    fp8Tflops: 17_500,
    bf16Tflops: 4_000,
    memoryBandwidthGBps: 16_250, // 130 TB/s / 8 GPUs (NVIDIA HGX)
    memoryCapacityGB: 288,
  },
  secondaryMetrics: {
    fp4Tflops: 9_000,
    fp8Tflops: 4_500,
    bf16Tflops: 2_250,
    memoryBandwidthGBps: 8_000,
    memoryCapacityGB: 192,
  },
  sources: [
    "https://www.nvidia.com/en-us/data-center/hgx/",
    "https://www.spheron.network/blog/nvidia-b200-complete-guide/",
  ],
};

export const VCPU_TARGETS: DeviceTargets = {
  primary: "AMD EPYC 9965 (Turin)",
  primaryMetrics: {
    cores: 192,
    threads: 384,
    l3CacheMB: 384,
    memoryBandwidthGBps: 614, // AMD up-to figure at DDR5-6400
    // ESTIMATE, not a vendor number: 192 cores x 2.25 GHz x 32 FP32
    // FLOPs/cycle (2x AVX-512 FMA). Label it wherever it is shown.
    fp32TflopsEstimate: 27.6,
  },
  sources: [
    "https://www.techpowerup.com/cpu-specs/epyc-9965.c3904",
    "https://www.cpu-monkey.com/en/cpu-amd_epyc_9965",
  ],
};

export const VRAM_TARGETS: DeviceTargets = {
  primary: "HBM3e (NVIDIA B200)",
  primaryMetrics: {
    memoryBandwidthGBps: 8_000,
    memoryCapacityGB: 192,
  },
  sources: ["https://www.spheron.network/blog/nvidia-b200-complete-guide/"],
};

export const VSSD_TARGETS: DeviceTargets = {
  primary: "Micron 9650 (PCIe Gen6)",
  primaryMetrics: {
    seqReadGBps: 28,
    seqWriteGBps: 14,
    randomReadIops: 5_500_000,
    randomWriteIops: 900_000,
  },
  sources: [
    "https://my.micron.com/products/storage/ssd/data-center-ssd/9650-ssd",
    "https://www.tomshardware.com/pc-components/ssds/microns-industry-first-pci-6-0-ssd-promises-sequential-reads-up-to-28-000-mb-s-245-tb-ssd-also-coming-for-those-who-need-capacity-more-than-cutting-edge-speed",
  ],
};

/**
 * vNPU targets (docs/VNPU.md §5.2, verified 2026-11-22). Primary:
 * Google TPU v6e (Trillium) — the current-generation inference-shaped
 * chip. Secondary: TPU v5p. Trainium2 (1,299 FP8 TFLOPS, 667 BF16,
 * 2,900 GB/s, 96 GiB, 224 MiB SBUF) and Inferentia2 (380 INT8 TOPS,
 * 190 BF16, 820 GiB/s, 32 GiB) are secondaries too, but DeviceTargets
 * holds one secondary — their rows are built with explicit targets in
 * bench-vnpu.ts. v5p INT8 918 TOPS is the Dec 2023 launch figure
 * (The Register), not the current docs page (which lists FP8 459);
 * both are recorded in VNPU.md §5.2.
 */
export const VNPU_TARGETS: DeviceTargets = {
  primary: "Google TPU v6e (Trillium)",
  secondary: "Google TPU v5p",
  primaryMetrics: {
    int8Tops: 1_836,
    bf16Tflops: 918,
    memoryBandwidthGBps: 1_638, // Google quotes v6e HBM in GB/s
    memoryCapacityGB: 32,
  },
  secondaryMetrics: {
    int8Tops: 918,
    bf16Tflops: 459,
    fp8Tflops: 459,
    memoryBandwidthGBps: 2_765, // Google quotes v5p HBM in GB/s
    memoryCapacityGB: 95,
  },
  sources: [
    "https://docs.cloud.google.com/tpu/docs/v6e",
    "https://docs.cloud.google.com/tpu/docs/v5p",
    "https://www.theregister.com/on-prem/2023/12/06/google-unveils-tpu-v5p-pods-to-accelerate-ai-training/",
    "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html",
    "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/inferentia2.html",
  ],
};
