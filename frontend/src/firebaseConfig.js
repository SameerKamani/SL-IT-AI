// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyC4M5EkfNcu5ftd1lZkYo0OArQ3mnPo14s",
  authDomain: "sl-it-ai.firebaseapp.com",
  projectId: "sl-it-ai",
  storageBucket: "sl-it-ai.firebasestorage.app",
  messagingSenderId: "457898425686",
  appId: "1:457898425686:web:39fa12be4937f4fb60aaf7",
  measurementId: "G-90NQYHMTG1"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);