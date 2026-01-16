/**
 * Data export utilities.
 */

/**
 * Export data as CSV file.
 */
export function exportToCSV(
  data: { headers: string[]; rows: (string | number)[][] },
  filename: string
): void {
  const csvContent = [
    data.headers.join(','),
    ...data.rows.map((row) => row.join(',')),
  ].join('\n');

  downloadFile(csvContent, filename, 'text/csv');
}

/**
 * Export data as JSON file.
 */
export function exportToJSON(
  data: unknown,
  filename: string,
  pretty = true
): void {
  const jsonContent = pretty
    ? JSON.stringify(data, null, 2)
    : JSON.stringify(data);

  downloadFile(jsonContent, filename, 'application/json');
}

/**
 * Export correlation data to CSV.
 */
export function exportCorrelationData(
  mcmcData?: { r: number[]; C_r: number[] },
  diffusionData?: { r: number[]; C_r: number[] },
  filename = 'correlation_data.csv'
): void {
  const headers = ['r'];
  const maxLength = Math.max(
    mcmcData?.r.length || 0,
    diffusionData?.r.length || 0
  );

  if (maxLength === 0) return;

  if (mcmcData) headers.push('MCMC_C_r');
  if (diffusionData) headers.push('Diffusion_C_r');

  const rows: (string | number)[][] = [];
  for (let i = 0; i < maxLength; i++) {
    const row: (string | number)[] = [
      mcmcData?.r[i] ?? diffusionData?.r[i] ?? '',
    ];
    if (mcmcData) row.push(mcmcData.C_r[i] ?? '');
    if (diffusionData) row.push(diffusionData.C_r[i] ?? '');
    rows.push(row);
  }

  exportToCSV({ headers, rows }, filename);
}

/**
 * Export distribution data to CSV.
 */
export function exportDistributionData(
  mcmcData?: { values: number[]; probabilities: number[] },
  diffusionData?: { values: number[]; probabilities: number[] },
  filename = 'distribution_data.csv'
): void {
  const headers = ['value'];
  const mcmcMap = new Map<number, number>();
  const diffusionMap = new Map<number, number>();

  if (mcmcData) {
    headers.push('MCMC_probability');
    mcmcData.values.forEach((v, i) => mcmcMap.set(v, mcmcData.probabilities[i]));
  }
  if (diffusionData) {
    headers.push('Diffusion_probability');
    diffusionData.values.forEach((v, i) =>
      diffusionMap.set(v, diffusionData.probabilities[i])
    );
  }

  const allValues = new Set([
    ...(mcmcData?.values || []),
    ...(diffusionData?.values || []),
  ]);

  const rows: (string | number)[][] = [...allValues]
    .sort((a, b) => a - b)
    .map((value) => {
      const row: (string | number)[] = [value];
      if (mcmcData) row.push(mcmcMap.get(value) ?? '');
      if (diffusionData) row.push(diffusionMap.get(value) ?? '');
      return row;
    });

  exportToCSV({ headers, rows }, filename);
}

/**
 * Download a file with the given content.
 */
function downloadFile(content: string, filename: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);

  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.style.display = 'none';

  document.body.appendChild(link);
  link.click();

  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
