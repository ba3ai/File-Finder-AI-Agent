import React from "react";

export const Button = ({ children, className = "", ...props }) => (
  <button
    className={`bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-xl ${className}`}
    {...props}
  >
    {children}
  </button>
);
