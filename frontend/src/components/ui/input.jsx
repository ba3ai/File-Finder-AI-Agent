import React from "react";

export const Input = ({ className = "", ...props }) => (
  <input
    className={`bg-zinc-800 border border-zinc-700 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-600 ${className}`}
    {...props}
  />
);
