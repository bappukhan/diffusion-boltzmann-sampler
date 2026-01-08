import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';

interface IsingVisualizerProps {
  spins: number[][] | null;
  size: number;
  title?: string;
}

export const IsingVisualizer: React.FC<IsingVisualizerProps> = ({
  spins,
  size,
  title = 'Spin Configuration',
}) => {
  const plotData = useMemo(() => {
    if (!spins) {
      // Empty placeholder
      return [
        {
          z: Array(size)
            .fill(null)
            .map(() => Array(size).fill(0)),
          type: 'heatmap' as const,
          colorscale: [
            [0, '#1e3a5f'],
            [0.5, '#1e3a5f'],
            [1, '#1e3a5f'],
          ],
          showscale: false,
        },
      ];
    }

    return [
      {
        z: spins,
        type: 'heatmap' as const,
        colorscale: [
          [0, '#3b82f6'], // Blue for -1 (spin down)
          [0.5, '#1e293b'], // Dark for 0 (shouldn't appear)
          [1, '#ef4444'], // Red for +1 (spin up)
        ],
        zmin: -1,
        zmax: 1,
        showscale: false,
        hoverongaps: false,
        hovertemplate: 'x: %{x}<br>y: %{y}<br>spin: %{z}<extra></extra>',
      },
    ];
  }, [spins, size]);

  const layout = useMemo(
    () => ({
      width: 350,
      height: 350,
      margin: { t: 30, b: 20, l: 20, r: 20 },
      title: {
        text: title,
        font: { size: 14, color: '#e2e8f0' },
      },
      xaxis: {
        visible: false,
        showgrid: false,
      },
      yaxis: {
        visible: false,
        showgrid: false,
        scaleanchor: 'x',
      },
      paper_bgcolor: 'transparent',
      plot_bgcolor: '#1e293b',
    }),
    [title]
  );

  return (
    <div className="bg-slate-800 rounded-lg p-4 shadow-lg">
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

export default IsingVisualizer;
