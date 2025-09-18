// FileSuggestionList.jsx
import React, { useState } from "react";

// Simple preview modal component
function FilePreviewModal({ file, onClose }) {
  if (!file) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-70">
      <div className="bg-white text-black p-6 rounded-lg w-4/5 h-4/5 relative">
        <button
          onClick={onClose}
          className="absolute top-2 right-4 text-xl font-bold"
        >
          ✖
        </button>
        <h2 className="text-lg font-semibold mb-4">{file.name}</h2>
        <iframe
          src={file.url}
          title={file.name}
          className="w-full h-full border rounded"
        />
      </div>
    </div>
  );
}

export default function FileSuggestionList({ files, onSelect, onSend }) {
  const [previewFile, setPreviewFile] = useState(null);

  return (
    <div className="bg-gray-800 p-4 rounded-lg shadow-lg max-w-xl mx-auto text-white space-y-4">
      <h2 className="text-xl font-semibold mb-4">📂 Suggested Files</h2>
      <ul className="space-y-2">
        {files.map((file, index) => (
          <li
            key={index}
            className="bg-gray-700 rounded-md p-3 hover:bg-gray-600 transition flex items-center justify-between"
          >
            <div className="space-y-1">
              <p className="font-medium">
                {index + 1}. {file.name}
              </p>
              <button
                className="text-sm text-blue-400 hover:underline"
                onClick={() => setPreviewFile(file)}
              >
                Preview
              </button>
            </div>
            <input
              type="checkbox"
              checked={file.selected}
              onChange={() => onSelect(index)}
              className="ml-4 w-5 h-5"
            />
          </li>
        ))}
      </ul>
      <button
        onClick={onSend}
        className="mt-4 w-full bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded"
      >
        ✅ Send Selected Files
      </button>

      {/* Render the modal */}
      <FilePreviewModal file={previewFile} onClose={() => setPreviewFile(null)} />
    </div>
  );
}
