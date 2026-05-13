import { useEffect, useState } from 'react';

type SnackbarTone = 'info' | 'success' | 'warning' | 'error';

interface SnackbarMessage {
  id: number;
  tone: SnackbarTone;
  message: string;
}

const EVENT_NAME = 'wmr-material-snackbar';

function emit(tone: SnackbarTone, message: string): void {
  window.dispatchEvent(
    new CustomEvent<SnackbarMessage>(EVENT_NAME, {
      detail: {
        id: Date.now() + Math.random(),
        tone,
        message,
      },
    }),
  );
}

export const notify = {
  info: (message: string) => emit('info', message),
  success: (message: string) => emit('success', message),
  warning: (message: string) => emit('warning', message),
  error: (message: string) => emit('error', message),
};

export function SnackbarHost() {
  const [message, setMessage] = useState<SnackbarMessage | null>(null);

  useEffect(() => {
    let timer: number | null = null;
    const onSnack = (event: Event) => {
      const next = (event as CustomEvent<SnackbarMessage>).detail;
      setMessage(next);
      if (timer !== null) {
        window.clearTimeout(timer);
      }
      timer = window.setTimeout(() => setMessage(null), next.tone === 'error' ? 5200 : 3600);
    };

    window.addEventListener(EVENT_NAME, onSnack);
    return () => {
      window.removeEventListener(EVENT_NAME, onSnack);
      if (timer !== null) {
        window.clearTimeout(timer);
      }
    };
  }, []);

  if (!message) return null;

  return (
    <div className={`md-snackbar md-snackbar-${message.tone}`} role="status" aria-live="polite">
      {message.message}
    </div>
  );
}
