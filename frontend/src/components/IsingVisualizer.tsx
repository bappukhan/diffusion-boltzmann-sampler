import React, { useMemo, useRef, useEffect, useState } from 'react';
import Plot from 'react-plotly.js';

type PlotSize = 'sm' | 'md' | 'lg' | 'auto';

interface IsingVisualizerProps {
  spins: number[][] | null;
  size: number;
  title?: string;
  /** Plot size: 'sm' (250px), 'md' (350px), 'lg' (450px), 'auto' (responsive) */
  plotSize?: PlotSize;
  /** Show spin value on hover */
  showHover?: boolean;
  /** Custom color for spin up (+1) */
  colorUp?: string;
  /** Custom color for spin down (-1) */
  colorDown?: string;
}

const plotSizes: Record<Exclude<PlotSize, 'auto'>, number> = {
  sm: 250,
  md: 350,
  lg: 450,
};

export const IsingVisualizer: React.FC<IsingVisualizerProps> = ({
  spins,
  size,
  title = 'Spin Configuration',
  plotSize = 'md',
  showHover = true,
  colorUp = '#ef4444',
  colorDown = '#3b82f6',
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: plotSizes.md, height: plotSizes.md });

  // Handle responsive sizing
  useEffect(() => {
    if (plotSize !== 'auto') {
      const pixelSize = plotSizes[plotSize];
      setDimensions({ width: pixelSize, height: pixelSize });
      return;
    }

    const updateDimensions = () => {
      if (containerRef.current) {
        const width = containerRef.current.clientWidth - 32; // Account for padding
        const constrainedSize = Math.min(Math.max(width, 200), 600);
        setDimensions({ width: constrainedSize, height: constrainedSize });
      }
    };

    updateDimensions();
    const observer = new ResizeObserver(updateDimensions);
    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => observer.disconnect();
  }, [plotSize]);

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
          [0, colorDown], // Blue for -1 (spin down)
          [0.5, '#1e293b'], // Dark for 0 (shouldn't appear)
          [1, colorUp], // Red for +1 (spin up)
        ],
        zmin: -1,
        zmax: 1,
        showscale: false,
        hoverongaps: false,
        hovertemplate: showHover
          ? 'x: %{x}<br>y: %{y}<br>spin: %{z}<extra></extra>'
          : '',
        hoverinfo: showHover ? undefined : ('none' as const),
      },
    ];
  }, [spins, size, colorUp, colorDown, showHover]);

  const layout = useMemo(
    () => ({
      width: dimensions.width,
      height: dimensions.height,
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
    [title, dimensions]
  );

  return (
    <div ref={containerRef} className="bg-slate-800 rounded-lg p-4 shadow-lg">
      <Plot
        data={plotData}
        layout={layout}
        config={{
          displayModeBar: false,
          responsive: plotSize === 'auto',
        }}
      />
    </div>
  );
};

export default IsingVisualizer;
