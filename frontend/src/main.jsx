import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

import { applyTheme, startThemeListener } from "./theme";
applyTheme();          // set theme on first paint
startThemeListener();  // react to OS theme + ProfileModal changes


ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
