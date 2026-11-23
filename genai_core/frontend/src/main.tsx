import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import './index.css'
import App from './App.tsx'

// Load the saved theme preference before React renders.
const savedTheme = localStorage.getItem('theme')

document.documentElement.dataset.theme =
  savedTheme === 'dark' ? 'dark' : 'light'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)