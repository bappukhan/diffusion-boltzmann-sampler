/**
 * Component tests for MagnetizationHistogram.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MagnetizationHistogram } from './MagnetizationHistogram';

// Mock Plotly to avoid canvas rendering issues in tests
vi.mock('react-plotly.js', () => ({
  default: ({ data, layout }: { data: unknown[]; layout: { title: { text: string } } }) => (
    <div data-testid="mock-plotly" data-traces={data.length}>
      <span>{layout.title?.text}</span>
    </div>
  ),
}));

// Mock export utilities
vi.mock('../utils/export', () => ({
  exportDistributionData: vi.fn(),
  exportCorrelationData: vi.fn(),
}));

describe('MagnetizationHistogram', () => {
  const mockMcmcData = {
    M: [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8],
    P_M: [0.1, 0.15, 0.1, 0.05, 0.02, 0.05, 0.1, 0.15, 0.1],
  };

  const mockDiffusionData = {
    M: [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8],
    P_M: [0.12, 0.14, 0.09, 0.04, 0.03, 0.04, 0.09, 0.14, 0.12],
  };

  describe('empty state', () => {
    it('should show empty message when no data', () => {
      render(<MagnetizationHistogram />);
      expect(screen.getByText(/Run comparison to see distribution/i)).toBeInTheDocument();
    });

    it('should show default title in empty state', () => {
      render(<MagnetizationHistogram />);
      expect(screen.getByText('Magnetization Distribution P(M)')).toBeInTheDocument();
    });
  });

  describe('with data', () => {
    it('should render plot with MCMC data', () => {
      render(<MagnetizationHistogram mcmcData={mockMcmcData} />);
      const plot = screen.getByTestId('mock-plotly');
      expect(plot).toBeInTheDocument();
      expect(plot.getAttribute('data-traces')).toBe('1');
    });

    it('should render plot with diffusion data', () => {
      render(<MagnetizationHistogram diffusionData={mockDiffusionData} />);
      const plot = screen.getByTestId('mock-plotly');
      expect(plot).toBeInTheDocument();
      expect(plot.getAttribute('data-traces')).toBe('1');
    });

    it('should render plot with both data sources', () => {
      render(
        <MagnetizationHistogram
          mcmcData={mockMcmcData}
          diffusionData={mockDiffusionData}
        />
      );
      const plot = screen.getByTestId('mock-plotly');
      expect(plot.getAttribute('data-traces')).toBe('2');
    });

    it('should use custom title', () => {
      render(
        <MagnetizationHistogram
          mcmcData={mockMcmcData}
          title="Custom P(M) Title"
        />
      );
      expect(screen.getByText('Custom P(M) Title')).toBeInTheDocument();
    });
  });

  describe('custom className', () => {
    it('should apply custom className', () => {
      const { container } = render(
        <MagnetizationHistogram
          mcmcData={mockMcmcData}
          className="custom-histogram-class"
        />
      );
      expect(container.firstChild).toHaveClass('custom-histogram-class');
    });
  });

  describe('data transformation', () => {
    it('should transform M/P_M to values/probabilities format', () => {
      // The component transforms the data internally to match DistributionPlot's API
      // Verify the transformation works by checking the plot renders
      render(
        <MagnetizationHistogram
          mcmcData={mockMcmcData}
          diffusionData={mockDiffusionData}
        />
      );
      const plot = screen.getByTestId('mock-plotly');
      expect(plot).toBeInTheDocument();
    });
  });
});
