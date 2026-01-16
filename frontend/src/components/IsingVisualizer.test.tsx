/**
 * Component tests for IsingVisualizer.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { IsingVisualizer } from './IsingVisualizer';

// Mock react-plotly.js
vi.mock('react-plotly.js', () => ({
  default: vi.fn(({ data, layout }: { data: unknown[]; layout: { title?: { text?: string } } }) => (
    <div data-testid="plotly-mock" data-title={layout.title?.text}>
      <div data-testid="plot-data">{JSON.stringify(data)}</div>
    </div>
  )),
}));

describe('IsingVisualizer', () => {
  const mockSpins = [
    [1, -1, 1],
    [-1, 1, -1],
    [1, -1, 1],
  ];

  describe('rendering', () => {
    it('should render with spins', () => {
      render(<IsingVisualizer spins={mockSpins} size={3} />);
      expect(screen.getByTestId('plotly-mock')).toBeInTheDocument();
    });

    it('should render without spins (null)', () => {
      render(<IsingVisualizer spins={null} size={3} />);
      expect(screen.getByTestId('plotly-mock')).toBeInTheDocument();
    });

    it('should render with custom title', () => {
      render(<IsingVisualizer spins={mockSpins} size={3} title="Custom Title" />);
      expect(screen.getByTestId('plotly-mock')).toHaveAttribute('data-title', 'Custom Title');
    });

    it('should render default title', () => {
      render(<IsingVisualizer spins={mockSpins} size={3} />);
      expect(screen.getByTestId('plotly-mock')).toHaveAttribute('data-title', 'Spin Configuration');
    });
  });

  describe('plot data', () => {
    it('should include spin data in plot', () => {
      render(<IsingVisualizer spins={mockSpins} size={3} />);
      const plotData = screen.getByTestId('plot-data');
      const data = JSON.parse(plotData.textContent || '[]');

      expect(data[0].z).toEqual(mockSpins);
      expect(data[0].type).toBe('heatmap');
    });

    it('should use correct color scale', () => {
      render(
        <IsingVisualizer
          spins={mockSpins}
          size={3}
          colorUp="#ff0000"
          colorDown="#0000ff"
        />
      );
      const plotData = screen.getByTestId('plot-data');
      const data = JSON.parse(plotData.textContent || '[]');

      expect(data[0].colorscale).toBeDefined();
      // First color (0) should be colorDown
      expect(data[0].colorscale[0][1]).toBe('#0000ff');
      // Last color (1) should be colorUp
      expect(data[0].colorscale[2][1]).toBe('#ff0000');
    });

    it('should set z range to -1 to 1', () => {
      render(<IsingVisualizer spins={mockSpins} size={3} />);
      const plotData = screen.getByTestId('plot-data');
      const data = JSON.parse(plotData.textContent || '[]');

      expect(data[0].zmin).toBe(-1);
      expect(data[0].zmax).toBe(1);
    });

    it('should show placeholder when spins is null', () => {
      render(<IsingVisualizer spins={null} size={3} />);
      const plotData = screen.getByTestId('plot-data');
      const data = JSON.parse(plotData.textContent || '[]');

      // Should have a z matrix of zeros
      expect(data[0].z).toHaveLength(3);
      expect(data[0].z[0]).toEqual([0, 0, 0]);
    });
  });

  describe('hover behavior', () => {
    it('should show hover by default', () => {
      render(<IsingVisualizer spins={mockSpins} size={3} />);
      const plotData = screen.getByTestId('plot-data');
      const data = JSON.parse(plotData.textContent || '[]');

      expect(data[0].hovertemplate).toContain('spin');
    });

    it('should hide hover when showHover is false', () => {
      render(<IsingVisualizer spins={mockSpins} size={3} showHover={false} />);
      const plotData = screen.getByTestId('plot-data');
      const data = JSON.parse(plotData.textContent || '[]');

      expect(data[0].hovertemplate).toBe('');
      expect(data[0].hoverinfo).toBe('none');
    });
  });

  describe('size props', () => {
    it('should accept size prop', () => {
      render(<IsingVisualizer spins={null} size={32} />);
      const plotData = screen.getByTestId('plot-data');
      const data = JSON.parse(plotData.textContent || '[]');

      expect(data[0].z).toHaveLength(32);
    });

    it('should accept plotSize prop', () => {
      const { container } = render(
        <IsingVisualizer spins={mockSpins} size={3} plotSize="sm" />
      );
      expect(container.querySelector('.bg-slate-800')).toBeInTheDocument();
    });
  });

  describe('styling', () => {
    it('should have slate background', () => {
      const { container } = render(<IsingVisualizer spins={mockSpins} size={3} />);
      expect(container.querySelector('.bg-slate-800')).toBeInTheDocument();
    });

    it('should have rounded corners', () => {
      const { container } = render(<IsingVisualizer spins={mockSpins} size={3} />);
      expect(container.querySelector('.rounded-lg')).toBeInTheDocument();
    });
  });
});
