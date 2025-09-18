import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { XCircle } from "lucide-react";

export default function DeleteConfirmModal({ isOpen, onClose, onConfirm, fileName }) {
  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          className="fixed inset-0 bg-black bg-opacity-50 backdrop-blur-sm z-50 flex items-center justify-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <motion.div
            className="bg-white rounded-lg shadow-lg max-w-md w-full p-6"
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 20 }}
          >
            {/* Warning Icon */}
            <div className="flex flex-col items-center text-center">
              <XCircle className="text-red-500 w-16 h-16 mb-3" strokeWidth={1.5} />
              <h2 className="text-xl font-semibold text-gray-800 mb-2">Are you sure?</h2>
              <p className="text-sm text-gray-600 mb-4">
                Do you really want to delete{" "}
                <span className="font-medium text-black">
                  {fileName || "this file"}
                </span>
                ?<br />
                This action cannot be undone.
              </p>
            </div>

            {/* Buttons Centered */}
            <div className="flex justify-center gap-4 mt-6">
              <button
                onClick={onClose}
                className="px-4 py-2 border border-gray-300 text-gray-700 rounded hover:bg-gray-100 transition"
              >
                Cancel
              </button>
              <button
                onClick={onConfirm}
                className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition"
              >
                Delete
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
