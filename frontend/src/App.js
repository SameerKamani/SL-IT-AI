import React, { useState, useEffect } from "react";
import Chatbot from "./Chatbot";
import Login from "./Login";
import { auth } from "./firebaseConfig";
import { onAuthStateChanged, signOut } from "firebase/auth";
import './App.css';

function App() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (firebaseUser) => {
      setUser(firebaseUser);
      setLoading(false);
    });
    return () => unsubscribe();
  }, []);

  const handleLogout = async () => {
    await signOut(auth);
    setUser(null);
  };

  const handleLoginSuccess = (user, rememberMe) => {
    setUser(user);
    // If remember me is checked, we could store additional state
    // For now, Firebase handles the persistence automatically
    if (rememberMe) {
      // You could store additional remember me state here if needed
      console.log("Remember me enabled for user:", user.email);
    }
  };

  if (loading) {
    return (
      <div className="App">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return <Login onLoginSuccess={handleLoginSuccess} />;
  }

  return (
    <div className="App">
      <h2 style={{marginTop: "1.5rem", textAlign: "center"}}>SL Helpdesk Chatbot</h2>
      <Chatbot username={user.email} userUid={user.uid} onLogout={handleLogout} />
    </div>
  );
}

export default App;