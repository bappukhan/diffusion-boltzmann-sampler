/**
 * Component tests for ControlPanel.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ControlPanel } from './ControlPanel';
import { useSimulationStore } from '../store/simulationStore';

// Reset store before each test
beforeEach(() => {
  useSimulationStore.getState().reset();
});

describe('ControlPanel', () => {
  const mockOnSample = vi.fn();
  const mockOnRandomize = vi.fn();

  beforeEach(() => {
    mockOnSample.mockClear();
    mockOnRandomize.mockClear();
  });

  describe('rendering', () => {
    it('should render Controls heading', () => {
      render(<ControlPanel onSample={mockOnSample} onRandomize={mockOnRandomize} />);
      expect(screen.getByText('Controls')).toBeInTheDocument();
    });

    it('should render temperature slider', () => {
      render(<ControlPanel onSample={mockOnSample} onRandomize={mockOnRandomize} />);
      expect(screen.getByText('Temperature')).toBeInTheDocument();
    });

    it('should render lattice size selector', () => {
      render(<ControlPanel onSample={mockOnSample} onRandomize={mockOnRandomize} />);
      expect(screen.getByText('Lattice Size')).toBeInTheDocument();
    });

    it('should render sampler buttons', () => {
      render(<ControlPanel onSample={mockOnSample} onRandomize={mockOnRandomize} />);
      expect(screen.getByText('MCMC')).toBeInTheDocument();
      expect(screen.getByText('Diffusion')).toBeInTheDocument();
    });

    it('should render action buttons', () => {
      render(<ControlPanel onSample={mockOnSample} onRandomize={mockOnRandomize} />);
      expect(screen.getByText('Generate Sample')).toBeInTheDocument();
      expect(screen.getByText('Random')).toBeInTheDocument();
    });
  });

  describe('temperature control', () => {
    it('should display current temperature', () => {
      render(<ControlPanel onSample={mockOnSample} onRandomize={mockOnRandomize} />);
      expect(screen.getByText(/T = 2.27/)).toBeInTheDocument();
    });

    it('should show critical temperature indicator when near T_c', () => {
      render(<ControlPanel onSample={mockOnSample} onRandomize={mockOnRandomize} />);
      expect(screen.getByText(/near T_c/)).toBeInTheDocument();
    });

    it('should update temperature when slider changes', () => {
      render(<ControlPanel onSample={mockOnSample} onRandomize={mockOnRandomize} />);
      const slider = screen.getByRole('slider', { name: /temperature/i });
      fireEvent.change(slider, { target: { value: '3.5' } });
      expect(useSimulationStore.getState().temperature).toBe(3.5);
    });

    it('should set temperature to T_c when clicking T_c label', () => {
      useSimulationStore.getState().setTemperature(1.0);
      render(<ControlPanel onSample={mockOnSample} onRandomize={mockOnRandomize} />);
      const tcLabel = screen.getByText(/T_c = 2.269/);
      fireEvent.click(tcLabel);
      expect(useSimulationStore.getState().temperature).toBeCloseTo(2.269, 3);
    });
  });

  describe('lattice size control', () => {
    it('should display current lattice size', () => {
      render(<ControlPanel onSample={mockOnSample} onRandomize={mockOnRandomize} />);
      expect(screen.getByText('32 x 32')).toBeInTheDocument();
    });

    it('should update lattice size when selection changes', () => {
      render(<ControlPanel onSample={mockOnSample} onRandomize={mockOnRandomize} />);
      const select = screen.getByRole('combobox');
      fireEvent.change(select, { target: { value: '64' } });
      expect(useSimulationStore.getState().latticeSize).toBe(64);
    });
  });

  describe('sampler type control', () => {
    it('should show MCMC as selected by default', () => {
      render(<ControlPanel onSample={mockOnSample} onRandomize={mockOnRandomize} />);
      const mcmcButton = screen.getByText('MCMC');
      expect(mcmcButton.className).toContain('bg-blue-600');
    });

    it('should switch to diffusion when clicked', () => {
      render(<ControlPanel onSample={mockOnSample} onRandomize={mockOnRandomize} />);
      fireEvent.click(screen.getByText('Diffusion'));
      expect(useSimulationStore.getState().samplerType).toBe('diffusion');
    });

    it('should switch back to MCMC when clicked', () => {
      useSimulationStore.getState().setSamplerType('diffusion');
      render(<ControlPanel onSample={mockOnSample} onRandomize={mockOnRandomize} />);
      fireEvent.click(screen.getByText('MCMC'));
      expect(useSimulationStore.getState().samplerType).toBe('mcmc');
    });
  });

  describe('animation steps control', () => {
    it('should display current steps count', () => {
      render(<ControlPanel onSample={mockOnSample} onRandomize={mockOnRandomize} />);
      expect(screen.getByText('100')).toBeInTheDocument();
    });

    it('should update steps when slider changes', () => {
      render(<ControlPanel onSample={mockOnSample} onRandomize={mockOnRandomize} />);
      const sliders = screen.getAllByRole('slider');
      const stepsSlider = sliders.find((s) => s.getAttribute('min') === '20');
      expect(stepsSlider).toBeDefined();
      fireEvent.change(stepsSlider!, { target: { value: '150' } });
      expect(useSimulationStore.getState().numSteps).toBe(150);
    });
  });

  describe('action buttons', () => {
    it('should call onSample when Generate Sample is clicked', () => {
      render(<ControlPanel onSample={mockOnSample} onRandomize={mockOnRandomize} />);
      fireEvent.click(screen.getByText('Generate Sample'));
      expect(mockOnSample).toHaveBeenCalledTimes(1);
    });

    it('should call onRandomize when Random is clicked', () => {
      render(<ControlPanel onSample={mockOnSample} onRandomize={mockOnRandomize} />);
      fireEvent.click(screen.getByText('Random'));
      expect(mockOnRandomize).toHaveBeenCalledTimes(1);
    });

    it('should disable buttons when running', () => {
      useSimulationStore.getState().setIsRunning(true);
      render(<ControlPanel onSample={mockOnSample} onRandomize={mockOnRandomize} />);

      const sampleButton = screen.getByText('Sampling...');
      expect(sampleButton).toBeDisabled();
    });

    it('should show Sampling... text when running', () => {
      useSimulationStore.getState().setIsRunning(true);
      render(<ControlPanel onSample={mockOnSample} onRandomize={mockOnRandomize} />);
      expect(screen.getByText('Sampling...')).toBeInTheDocument();
    });
  });

  describe('reset to defaults', () => {
    it('should not show reset button when config is default', () => {
      render(<ControlPanel onSample={mockOnSample} onRandomize={mockOnRandomize} />);
      expect(screen.queryByText('Reset to Defaults')).not.toBeInTheDocument();
    });

    it('should show reset button when config differs from default', () => {
      useSimulationStore.getState().setTemperature(1.5);
      render(<ControlPanel onSample={mockOnSample} onRandomize={mockOnRandomize} />);
      expect(screen.getByText('Reset to Defaults')).toBeInTheDocument();
    });

    it('should reset config when reset button is clicked', () => {
      useSimulationStore.getState().setTemperature(1.5);
      useSimulationStore.getState().setLatticeSize(64);
      render(<ControlPanel onSample={mockOnSample} onRandomize={mockOnRandomize} />);

      fireEvent.click(screen.getByText('Reset to Defaults'));

      const state = useSimulationStore.getState();
      expect(state.temperature).toBeCloseTo(2.27, 2);
      expect(state.latticeSize).toBe(32);
    });
  });

  describe('current state display', () => {
    it('should not show state when energy and magnetization are null', () => {
      render(<ControlPanel onSample={mockOnSample} onRandomize={mockOnRandomize} />);
      expect(screen.queryByText('Current State')).not.toBeInTheDocument();
    });

    it('should show energy when set', () => {
      useSimulationStore.getState().setEnergy(-1.234);
      render(<ControlPanel onSample={mockOnSample} onRandomize={mockOnRandomize} />);
      expect(screen.getByText('Current State')).toBeInTheDocument();
      expect(screen.getByText('Energy/spin')).toBeInTheDocument();
      expect(screen.getByText('-1.234')).toBeInTheDocument();
    });

    it('should show magnetization when set', () => {
      useSimulationStore.getState().setMagnetization(0.567);
      render(<ControlPanel onSample={mockOnSample} onRandomize={mockOnRandomize} />);
      expect(screen.getByText('Magnetization')).toBeInTheDocument();
      expect(screen.getByText('0.567')).toBeInTheDocument();
    });
  });
});
