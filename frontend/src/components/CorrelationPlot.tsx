import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { exportCorrelationData, exportDistributionData } from '../utils/export';

interface CorrelationPlotProps {
  mcmcData?: { r: number[]; C_r: number[] };
  diffusionData?: { r: number[]; C_r: number[] };
  title?: string;
  /** Show export button */
  showExport?: boolean;
}

export const CorrelationPlot: React.FC<CorrelationPlotProps> = ({
  mcmcData,
  diffusionData,
  title = 'Spin-Spin Correlation',
  showExport = true,
}) => {
  const handleExport = () => {
    exportCorrelationData(mcmcData, diffusionData, 'correlation_data.csv');
  };
  const plotData = useMemo(() => {
    const traces = [];

    if (mcmcData && mcmcData.r.length > 0) {
      traces.push({
        x: mcmcData.r,
        y: mcmcData.C_r,
        name: 'MCMC',
        type: 'scatter' as const,
        mode: 'lines+markers' as const,
        marker: { color: '#3b82f6', size: 6 },
        line: { color: '#3b82f6', width: 2 },
      });
    }

    if (diffusionData && diffusionData.r.length > 0) {
      traces.push({
        x: diffusionData.r,
        y: diffusionData.C_r,
        name: 'Diffusion',
        type: 'scatter' as const,
        mode: 'lines+markers' as const,
        marker: { color: '#a855f7', size: 6 },
        line: { color: '#a855f7', width: 2 },
      });
    }

    return traces;
  }, [mcmcData, diffusionData]);

  const layout = useMemo(
    () => ({
      width: 400,
      height: 300,
      margin: { t: 40, b: 50, l: 60, r: 30 },
      title: {
        text: title,
        font: { size: 14, color: '#e2e8f0' },
      },
      xaxis: {
        title: { text: 'Distance r', font: { size: 12, color: '#94a3b8' } },
        color: '#64748b',
        gridcolor: '#334155',
        zerolinecolor: '#475569',
      },
      yaxis: {
        title: { text: 'C(r)', font: { size: 12, color: '#94a3b8' } },
        color: '#64748b',
        gridcolor: '#334155',
        zerolinecolor: '#475569',
      },
      paper_bgcolor: 'transparent',
      plot_bgcolor: '#1e293b',
      legend: {
        x: 0.7,
        y: 0.95,
        bgcolor: 'rgba(30, 41, 59, 0.8)',
        font: { color: '#e2e8f0' },
      },
      showlegend: plotData.length > 1,
    }),
    [title, plotData.length]
  );

  if (plotData.length === 0) {
    return (
      <div className="bg-slate-800 rounded-lg p-4 shadow-lg">
        <h3 className="text-sm font-medium text-slate-400 mb-3">{title}</h3>
        <div className="text-center text-slate-500 py-8 h-[250px] flex items-center justify-center">
          Run comparison to see correlation data
        </div>
      </div>
    );
  }

  return (
    <div className="bg-slate-800 rounded-lg p-4 shadow-lg relative">
      {showExport && plotData.length > 0 && (
        <button
          onClick={handleExport}
          className="absolute top-2 right-2 p-1.5 text-slate-400 hover:text-white hover:bg-slate-700 rounded transition-colors"
          title="Export data as CSV"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
        </button>
      )}
      <Plot
        data={plotData}
        layout={layout}
        config={{
          displayModeBar: false,
          responsive: true,
        }}
      />
    </div>
  );
};

interface DistributionPlotProps {
  mcmcData?: { values: number[]; probabilities: number[] };
  diffusionData?: { values: number[]; probabilities: number[] };
  title?: string;
  xlabel?: string;
  /** Show export button */
  showExport?: boolean;
}

export const DistributionPlot: React.FC<DistributionPlotProps> = ({
  mcmcData,
  diffusionData,
  title = 'Distribution',
  xlabel = 'Value',
  showExport = true,
}) => {
  const handleExport = () => {
    const filename = title.toLowerCase().replace(/\s+/g, '_') + '_data.csv';
    exportDistributionData(mcmcData, diffusionData, filename);
  };
  const plotData = useMemo(() => {
    const traces = [];

    if (mcmcData && mcmcData.values.length > 0) {
      traces.push({
        x: mcmcData.values,
        y: mcmcData.probabilities,
        name: 'MCMC',
        type: 'bar' as const,
        marker: { color: 'rgba(59, 130, 246, 0.7)' },
      });
    }

    if (diffusionData && diffusionData.values.length > 0) {
      traces.push({
        x: diffusionData.values,
        y: diffusionData.probabilities,
        name: 'Diffusion',
        type: 'bar' as const,
        marker: { color: 'rgba(168, 85, 247, 0.7)' },
      });
    }

    return traces;
  }, [mcmcData, diffusionData]);

  const layout = useMemo(
    () => ({
      width: 400,
      height: 300,
      margin: { t: 40, b: 50, l: 60, r: 30 },
      title: {
        text: title,
        font: { size: 14, color: '#e2e8f0' },
      },
      xaxis: {
        title: { text: xlabel, font: { size: 12, color: '#94a3b8' } },
        color: '#64748b',
        gridcolor: '#334155',
      },
      yaxis: {
        title: { text: 'P', font: { size: 12, color: '#94a3b8' } },
        color: '#64748b',
        gridcolor: '#334155',
      },
      paper_bgcolor: 'transparent',
      plot_bgcolor: '#1e293b',
      barmode: 'overlay' as const,
      legend: {
        x: 0.7,
        y: 0.95,
        bgcolor: 'rgba(30, 41, 59, 0.8)',
        font: { color: '#e2e8f0' },
      },
      showlegend: plotData.length > 1,
    }),
    [title, xlabel, plotData.length]
  );

  if (plotData.length === 0) {
    return (
      <div className="bg-slate-800 rounded-lg p-4 shadow-lg">
        <h3 className="text-sm font-medium text-slate-400 mb-3">{title}</h3>
        <div className="text-center text-slate-500 py-8 h-[250px] flex items-center justify-center">
          Run comparison to see distribution
        </div>
      </div>
    );
  }

  return (
    <div className="bg-slate-800 rounded-lg p-4 shadow-lg relative">
      {showExport && plotData.length > 0 && (
        <button
          onClick={handleExport}
          className="absolute top-2 right-2 p-1.5 text-slate-400 hover:text-white hover:bg-slate-700 rounded transition-colors"
          title="Export data as CSV"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
        </button>
      )}
      <Plot
        data={plotData}
        layout={layout}
        config={{
          displayModeBar: false,
          responsive: true,
        }}
      />
    </div>
  );
};

export default CorrelationPlot;
