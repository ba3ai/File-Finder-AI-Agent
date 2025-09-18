// FilePreviewModal.jsx
import React from "react";

const FilePreviewModal = ({ show, onClose, fileUrl, fileName }) => {
  if (!show) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-70">
      <div className="bg-white p-6 rounded-lg w-4/5 h-4/5 relative">
        <button
          onClick={onClose}
          className="absolute top-2 right-4 text-xl font-bold"
        >
          ✖
        </button>
        <h2 className="text-lg font-semibold mb-4">{fileName}</h2>
        <iframe
          src={fileUrl}
          title="Preview"
          className="w-full h-full border"
        />
      </div>
    </div>
  );
};

export default FilePreviewModal;
