import {
  type ButtonHTMLAttributes,
  type CSSProperties,
  type MouseEventHandler,
  type ReactNode,
  createElement,
  useEffect,
  useRef,
} from 'react';

import '@material/web/button/elevated-button.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/filled-tonal-button.js';
import '@material/web/button/outlined-button.js';
import '@material/web/button/text-button.js';
import '@material/web/checkbox/checkbox.js';
import '@material/web/dialog/dialog.js';
import '@material/web/progress/linear-progress.js';
import '@material/web/radio/radio.js';
import '@material/web/select/outlined-select.js';
import '@material/web/select/select-option.js';
import '@material/web/slider/slider.js';
import '@material/web/switch/switch.js';
import '@material/web/textfield/outlined-text-field.js';
import 'material-symbols/rounded.css';

type CustomElementProps = Record<string, unknown>;

function setElementProperty(element: HTMLElement | null, key: string, value: unknown): void {
  if (element) {
    (element as unknown as Record<string, unknown>)[key] = value;
  }
}

export function MaterialIcon({ name, className = '' }: { name: string; className?: string }) {
  return (
    <span className={`material-symbols-rounded md-icon ${className}`} aria-hidden="true">
      {name}
    </span>
  );
}

interface MdButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'type'> {
  variant?: 'filled' | 'tonal' | 'outlined' | 'text' | 'elevated';
  tone?: 'default' | 'danger' | 'success';
  icon?: string;
  trailingIcon?: string;
  loading?: boolean;
}

export function MdButton({
  variant = 'tonal',
  tone = 'default',
  icon,
  trailingIcon,
  loading = false,
  disabled,
  className = '',
  children,
  ...rest
}: MdButtonProps) {
  const tag = {
    elevated: 'md-elevated-button',
    filled: 'md-filled-button',
    outlined: 'md-outlined-button',
    text: 'md-text-button',
    tonal: 'md-filled-tonal-button',
  }[variant];
  const props: CustomElementProps = {
    ...rest,
    disabled: disabled || loading ? true : undefined,
    class: `md-button md-button-${tone} ${className}`.trim(),
  };

  return createElement(
    tag,
    props,
    loading ? <span className="md-button-spinner" aria-hidden="true" /> : null,
    icon ? <MaterialIcon name={icon} /> : null,
    <span>{children}</span>,
    trailingIcon ? <MaterialIcon name={trailingIcon} /> : null,
  );
}

interface MdIconButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'type' | 'children'> {
  icon: string;
  label: string;
  selected?: boolean;
  disabled?: boolean;
  className?: string;
  onClick?: MouseEventHandler<HTMLButtonElement>;
}

export function MdIconButton({
  icon,
  label,
  selected = false,
  disabled = false,
  className = '',
  onClick,
  ...rest
}: MdIconButtonProps) {
  return (
    <button
      {...rest}
      type="button"
      aria-label={label}
      aria-pressed={selected ? true : undefined}
      disabled={disabled}
      className={`md-icon-button ${selected ? 'is-selected' : ''} ${className}`.trim()}
      onClick={onClick}
    >
      <MaterialIcon name={icon} />
    </button>
  );
}

export interface MdSelectOption<T extends string> {
  value: T;
  label: ReactNode;
  disabled?: boolean;
}

interface MdSelectProps<T extends string> {
  value: T;
  label?: string;
  options: MdSelectOption<T>[];
  disabled?: boolean;
  className?: string;
  onChange: (value: T) => void;
}

export function MdSelect<T extends string>({
  value,
  label,
  options,
  disabled = false,
  className = '',
  onChange,
}: MdSelectProps<T>) {
  const ref = useRef<HTMLElement | null>(null);

  useEffect(() => {
    setElementProperty(ref.current, 'value', value);
  }, [value]);

  const handleInput = (event: Event) => {
    const target = event.currentTarget as HTMLElement & { value?: string };
    onChange(String(target.value || value) as T);
  };

  return (
    <md-outlined-select
      ref={ref}
      class={`md-select ${className}`.trim()}
      label={label}
      value={value}
      disabled={disabled || undefined}
      onInput={handleInput as never}
    >
      {options.map((option) => (
        <md-select-option key={option.value} value={option.value} disabled={option.disabled || undefined}>
          <div slot="headline">{option.label}</div>
        </md-select-option>
      ))}
    </md-outlined-select>
  );
}

interface MdSwitchProps {
  checked: boolean;
  disabled?: boolean;
  className?: string;
  label?: string;
  onChange: (checked: boolean) => void;
}

export function MdSwitch({
  checked,
  disabled = false,
  className = '',
  label,
  onChange,
}: MdSwitchProps) {
  const ref = useRef<HTMLElement | null>(null);

  useEffect(() => {
    setElementProperty(ref.current, 'selected', checked);
    setElementProperty(ref.current, 'disabled', disabled);
  }, [checked, disabled]);

  const handleInput = (event: Event) => {
    const target = event.currentTarget as HTMLElement & { selected?: boolean };
    onChange(!!target.selected);
  };

  return (
    <md-switch
      ref={ref}
      aria-label={label}
      class={`md-switch ${className}`.trim()}
      selected={checked || undefined}
      disabled={disabled || undefined}
      onInput={handleInput as never}
    />
  );
}

interface MdSliderProps {
  min: number;
  max: number;
  value: number;
  step?: number;
  disabled?: boolean;
  className?: string;
  ariaLabel?: string;
  onChange: (value: number) => void;
}

export function MdSlider({
  min,
  max,
  value,
  step = 1,
  disabled = false,
  className = '',
  ariaLabel,
  onChange,
}: MdSliderProps) {
  const ref = useRef<HTMLElement | null>(null);

  useEffect(() => {
    setElementProperty(ref.current, 'value', value);
  }, [value]);

  const handleInput = (event: Event) => {
    const target = event.currentTarget as HTMLElement & { value?: number | string };
    onChange(Number(target.value));
  };

  return (
    <md-slider
      ref={ref}
      aria-label={ariaLabel}
      min={min}
      max={max}
      value={value}
      step={step}
      disabled={disabled || undefined}
      class={`md-slider ${className}`.trim()}
      onInput={handleInput as never}
    />
  );
}

interface MdTextFieldProps {
  value: string;
  label?: string;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
  onChange: (value: string) => void;
}

export function MdTextField({
  value,
  label,
  placeholder,
  disabled = false,
  className = '',
  onChange,
}: MdTextFieldProps) {
  const ref = useRef<HTMLElement | null>(null);

  useEffect(() => {
    setElementProperty(ref.current, 'value', value);
  }, [value]);

  const handleInput = (event: Event) => {
    const target = event.currentTarget as HTMLElement & { value?: string };
    onChange(String(target.value || ''));
  };

  return (
    <md-outlined-text-field
      ref={ref}
      class={`md-text-field ${className}`.trim()}
      label={label}
      placeholder={placeholder}
      value={value}
      disabled={disabled || undefined}
      onInput={handleInput as never}
    />
  );
}

export function MdLinearProgress({
  value,
  className = '',
}: {
  value: number;
  className?: string;
}) {
  const ref = useRef<HTMLElement | null>(null);
  const normalized = Math.max(0, Math.min(1, Number(value) || 0));

  useEffect(() => {
    setElementProperty(ref.current, 'value', normalized);
  }, [normalized]);

  return <md-linear-progress ref={ref} class={`md-linear-progress ${className}`.trim()} value={normalized} />;
}

export function MdChip({
  children,
  tone = 'neutral',
  className = '',
}: {
  children: ReactNode;
  tone?: 'neutral' | 'primary' | 'success' | 'warning' | 'danger';
  className?: string;
}) {
  return <span className={`md-chip md-chip-${tone} ${className}`.trim()}>{children}</span>;
}

export function MdSurface({
  children,
  className = '',
  style,
}: {
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
}) {
  return (
    <section className={`md-surface ${className}`.trim()} style={style}>
      {children}
    </section>
  );
}

export function MdEmptyState({
  icon,
  title,
  description,
  action,
  className = '',
}: {
  icon: string;
  title: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={`md-empty-state ${className}`.trim()}>
      <span className="md-empty-state-icon">
        <MaterialIcon name={icon} />
      </span>
      <div className="md-empty-state-copy">
        <h3>{title}</h3>
        {description ? <p>{description}</p> : null}
      </div>
      {action ? <div className="md-empty-state-action">{action}</div> : null}
    </div>
  );
}

export function MdStatusMetric({
  label,
  value,
  tone = 'neutral',
  className = '',
}: {
  label: ReactNode;
  value: ReactNode;
  tone?: 'neutral' | 'primary' | 'success' | 'warning' | 'danger';
  className?: string;
}) {
  return (
    <div className={`md-status-metric md-status-metric-${tone} ${className}`.trim()}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export function MdTaskPanel({
  icon,
  title,
  subtitle,
  children,
  footer,
  className = '',
}: {
  icon?: string;
  title: ReactNode;
  subtitle?: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
  className?: string;
}) {
  return (
    <section className={`md-task-panel ${className}`.trim()}>
      <header className="md-task-panel-header">
        {icon ? (
          <span className="md-task-panel-icon">
            <MaterialIcon name={icon} />
          </span>
        ) : null}
        <div>
          <h3>{title}</h3>
          {subtitle ? <p>{subtitle}</p> : null}
        </div>
      </header>
      <div className="md-task-panel-body">{children}</div>
      {footer ? <footer className="md-task-panel-footer">{footer}</footer> : null}
    </section>
  );
}

export function MdInspectorList({
  children,
  className = '',
}: {
  children: ReactNode;
  className?: string;
}) {
  return <div className={`md-inspector-list ${className}`.trim()}>{children}</div>;
}

export function MdInspectorRow({
  label,
  value,
  action,
  selected = false,
  onClick,
}: {
  label: ReactNode;
  value: ReactNode;
  action?: ReactNode;
  selected?: boolean;
  onClick?: () => void;
}) {
  const content = (
    <>
      <span className="md-inspector-row-label">{label}</span>
      <span className="md-inspector-row-value">{value}</span>
      {action ? <span className="md-inspector-row-action">{action}</span> : null}
    </>
  );

  return (
    <div
      className={`md-inspector-row ${selected ? 'is-selected' : ''}`.trim()}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onClick={onClick}
      onKeyDown={(event) => {
        if (!onClick) return;
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          onClick();
        }
      }}
    >
      {content}
    </div>
  );
}
