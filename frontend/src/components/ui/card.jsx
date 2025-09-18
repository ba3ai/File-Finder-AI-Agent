import React from "react";

export const Card = ({ children }) => (
  <div className="bg-zinc-900 rounded-2xl shadow-md p-4">{children}</div>
);

export const CardContent = ({ children, className = "" }) => (
  <div className={`text-sm text-white ${className}`}>{children}</div>
);
