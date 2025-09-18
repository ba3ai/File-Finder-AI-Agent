// src/components/UnauthorizedPage.jsx
import React from "react";
import { useNavigate } from "react-router-dom";

export default function UnauthorizedPage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100">
      <div className="bg-white shadow-lg rounded-xl p-10 max-w-md text-center">
        <div className="text-5xl text-red-500 mb-4">❌</div>
        <h1 className="text-2xl font-bold text-gray-800 mb-2">Unauthorized Domain</h1>
        <p className="text-gray-600 mb-6">
          You do not have access to this application with your current email domain.
        </p>
        <button
          onClick={() => navigate("/")}
          className="px-6 py-2 bg-gray-800 text-white rounded-md hover:bg-gray-700 transition"
        >
          Go Back Home
        </button>
      </div>
    </div>
  );
}
