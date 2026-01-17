/**
 * Magnetization distribution histogram component.
 *
 * Specialized wrapper around DistributionPlot for P(M) visualization.
 * Shows magnetization distribution from MCMC and/or diffusion samples.
 */

import React from 'react';
import { DistributionPlot } from './CorrelationPlot';

interface MagnetizationHistogramProps {
  /** MCMC magnetization distribution data */
  mcmcData?: { M: number[]; P_M: number[] };
  /** Diffusion magnetization distribution data */
  diffusionData?: { M: number[]; P_M: number[] };
  /** Plot title */
  title?: string;
  /** Show export button */
  showExport?: boolean;
  /** Custom CSS class */
  className?: string;
}

/**
 * MagnetizationHistogram component for visualizing P(M).
 *
 * Below T_c (~2.27), expect bimodal distribution (spontaneous magnetization).
 * Above T_c, expect unimodal distribution centered at M=0.
 */
export const MagnetizationHistogram: React.FC<MagnetizationHistogramProps> = ({
  mcmcData,
  diffusionData,
  title = 'Magnetization Distribution P(M)',
  showExport = true,
  className,
}) => {
  // Transform data to DistributionPlot format
  const mcmcPlotData = mcmcData
    ? { values: mcmcData.M, probabilities: mcmcData.P_M }
    : undefined;

  const diffusionPlotData = diffusionData
    ? { values: diffusionData.M, probabilities: diffusionData.P_M }
    : undefined;

  return (
    <div className={className}>
      <DistributionPlot
        mcmcData={mcmcPlotData}
        diffusionData={diffusionPlotData}
        title={title}
        xlabel="Magnetization M"
        showExport={showExport}
      />
    </div>
  );
};

export default MagnetizationHistogram;
