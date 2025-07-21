import React, { useState } from "react";
import { auth } from "./firebaseConfig";
import { signInWithEmailAndPassword } from "firebase/auth";
import "./App.css";
import "./Login.css";

function Login({ onLoginSuccess }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [rememberMe, setRememberMe] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const userCredential = await signInWithEmailAndPassword(auth, email, password);
      if (onLoginSuccess) onLoginSuccess(userCredential.user, rememberMe);
    } catch (err) {
      setError("Login failed: " + (err?.message || "Unknown error"));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-bg">
      <div className="login-card">
        <h2 className="login-title">Login to SL Helpdesk</h2>
        <form className="login-form" onSubmit={handleLogin}>
          <div className="login-field">
            <label htmlFor="email" className="login-label">Email</label>
            <input
              id="email"
              type="email"
              className="login-input"
              value={email}
              onChange={e => setEmail(e.target.value)}
              placeholder="Enter your email"
              autoComplete="username"
              required
              disabled={loading}
            />
          </div>
          <div className="login-field">
            <label htmlFor="password" className="login-label">Password</label>
            <input
              id="password"
              type="password"
              className="login-input"
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder="Enter your password"
              autoComplete="current-password"
              required
              disabled={loading}
            />
          </div>
          <div className="remember-me-container">
            <label className="remember-me-label">
              <input
                type="checkbox"
                className="remember-me-checkbox"
                checked={rememberMe}
                onChange={e => setRememberMe(e.target.checked)}
                disabled={loading}
              />
              <span className="remember-me-text">Remember me</span>
            </label>
          </div>
          {error && <div className="login-error">{error}</div>}
          <button
            type="submit"
            className="login-btn"
            disabled={loading}
          >
            {loading ? "Logging in..." : "Login"}
          </button>
        </form>
      </div>
    </div>
  );
}

export default Login; 