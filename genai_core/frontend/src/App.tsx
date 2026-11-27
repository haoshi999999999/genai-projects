import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
} from 'react-router-dom'

import type { ReactNode } from 'react'

import LoginPage from './pages/LoginPage'
import RegisterPage from './pages/RegisterPage'
import ChatPage from './pages/ChatPage'
import DocumentsPage from './pages/DocumentsPage'
import SettingsPage from './pages/SettingsPage'

// Protect pages that require authentication
function ProtectedRoute({ children }: { children: ReactNode }) {
  const token = sessionStorage.getItem('access_token')

  if (!token) {
    return <Navigate to="/" replace />
  }

  return children
}

function App() {
  return (
    <BrowserRouter>
      <Routes>

        {/* Public Login Page */}
        <Route path="/" element={<LoginPage />} />

        {/* Public Registration Page */}
        <Route path="/register" element={<RegisterPage />} />

        {/* Protected Chat Page */}
        <Route
          path="/chat"
          element={
            <ProtectedRoute>
              <ChatPage />
            </ProtectedRoute>
          }
        />

        {/* Protected Documents Page */}
        <Route
          path="/documents"
          element={
            <ProtectedRoute>
              <DocumentsPage />
            </ProtectedRoute>
          }
        />

        {/* Protected Settings Page */}
        <Route
          path="/settings"
          element={
            <ProtectedRoute>
              <SettingsPage />
            </ProtectedRoute>
          }
        />

        {/* Redirect unknown URLs */}
        <Route path="*" element={<Navigate to="/" replace />} />

      </Routes>
    </BrowserRouter>
  )
}

export default App