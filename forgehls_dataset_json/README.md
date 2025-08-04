# ForgeHLS Dataset JSON Format

This directory contains the ForgeHLS dataset in JSON format, including both the original and compressed versions.

## File Information

### Compression Results

- Original file: `data_of_designs_forgehls_with_strategy_formatted.json`
  - Size: 924MB
- Compressed file: `data_of_designs_forgehls_with_strategy_formatted.json.gz`
  - Size: 24MB

### Example Entry

```json
{
    "File Path": "/root/code/HLSBatchProcessor-main/data/designs/CHStone/adpcm/design_626/project/solution1/syn/report/csynth.xml",
    "Part": "xcu280-fsvh2892-2L-e",
    "TargetClockPeriod": 10,
    "Best-caseLatency": 11485,
    "Worst-caseLatency": 15735,
    "BRAM_18K": 0,
    "LUT": 10365,
    "DSP": 44,
    "FF": 2750,
    "Avialable_BRAM_18K": 4032,
    "Avialable_LUT": 1303680,
    "Avialable_DSP": 9024,
    "Avialable_FF": 2607360,
    "ResourceMetric": 0.003470291,
    "design_id": "design_626",
    "algo_name": "adpcm",
    "source_name": "CHStone",
    "source_code": [
        {
            "file_name": "adpcm.c",
            "file_content": "/* CHStone benchmark code... */"
        }
    ],
    "code_length": 25961,
    "pragma_number": 32,
    "is_pareto": true,
    "is_kernel": false,
    "top_function_name": "main",
    "latency-resource-strategy": "low-latency-high-resource"
}
```


### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `File Path` | string | Path to the synthesis report XML file |
| `Part` | string | Target FPGA device (e.g., "xcu280-fsvh2892-2L-e") |
| `TargetClockPeriod` | number | Target clock period in nanoseconds |
| `Best-caseLatency` | number | Best-case latency in clock cycles |
| `Worst-caseLatency` | number | Worst-case latency in clock cycles |
| `BRAM_18K` | number | Block RAM usage (18K blocks) |
| `LUT` | number | Look-Up Table usage |
| `DSP` | number | Digital Signal Processor usage |
| `FF` | number | Flip-Flop usage |
| `Avialable_*` | number | Available resources on the target device |
| `ResourceMetric` | number | Calculated resource efficiency metric |
| `design_id` | string | Unique identifier for the design variant |
| `algo_name` | string | Algorithm name (e.g., "adpcm", "aes", etc.) |
| `source_name` | string | Source benchmark suite (e.g., "CHStone") |
| `source_code` | array | Array of source code files with content |
| `code_length` | number | Total length of source code |
| `pragma_number` | number | Number of HLS pragmas used |
| `is_pareto` | boolean | Whether design is on Pareto frontier |
| `is_kernel` | boolean | Whether this is a kernel function |
| `top_function_name` | string | Name of the top-level function |
| `latency-resource-strategy` | string | Optimization strategy used |
