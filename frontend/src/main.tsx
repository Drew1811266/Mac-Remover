// 前端入口文件：
// 1) 挂载根组件 App
// 2) 引入全局样式
import React from 'react';
import ReactDOM from 'react-dom/client';

import App from './App';
import './styles.css';

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  // StrictMode 只在开发期提供额外检查，不影响生产功能逻辑。
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
