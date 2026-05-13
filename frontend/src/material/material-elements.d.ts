import type { DetailedHTMLProps, HTMLAttributes } from 'react';

type MaterialElementProps = DetailedHTMLProps<HTMLAttributes<HTMLElement>, HTMLElement> & {
  checked?: boolean;
  class?: string;
  disabled?: boolean;
  label?: string;
  max?: number;
  min?: number;
  open?: boolean;
  placeholder?: string;
  selected?: boolean;
  step?: number;
  value?: number | string;
};

declare global {
  namespace JSX {
    interface IntrinsicElements {
      'md-checkbox': MaterialElementProps;
      'md-dialog': MaterialElementProps;
      'md-icon-button': MaterialElementProps;
      'md-linear-progress': MaterialElementProps;
      'md-outlined-select': MaterialElementProps;
      'md-outlined-text-field': MaterialElementProps;
      'md-radio': MaterialElementProps;
      'md-select-option': MaterialElementProps;
      'md-slider': MaterialElementProps;
      'md-switch': MaterialElementProps;
    }
  }
}

export {};
