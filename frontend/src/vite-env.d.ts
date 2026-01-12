/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string;
  readonly VITE_WS_BASE_URL: string;
  readonly VITE_HEALTH_CHECK_INTERVAL: string;
  readonly VITE_DEFAULT_LATTICE_SIZE: string;
  readonly VITE_DEFAULT_TEMPERATURE: string;
  readonly VITE_MAX_LATTICE_SIZE: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

declare module 'react-plotly.js' {
  import { Component } from 'react';
  import Plotly from 'plotly.js';

  interface PlotParams {
    data: Plotly.Data[];
    layout?: Partial<Plotly.Layout>;
    config?: Partial<Plotly.Config>;
    style?: React.CSSProperties;
    className?: string;
    useResizeHandler?: boolean;
    onInitialized?: (figure: Readonly<Plotly.Figure>, graphDiv: HTMLElement) => void;
    onUpdate?: (figure: Readonly<Plotly.Figure>, graphDiv: HTMLElement) => void;
    onPurge?: (figure: Readonly<Plotly.Figure>, graphDiv: HTMLElement) => void;
    onError?: (error: Error) => void;
    divId?: string;
  }

  export default class Plot extends Component<PlotParams> {}
}
