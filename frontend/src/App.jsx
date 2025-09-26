// src/App.jsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import HeaderModeMenu from "./components/HeaderModeMenu.jsx";

// Components
import ChatPanel from "./components/ChatPanel.jsx";
import ChatInput from "./components/ChatInput.jsx";
import WelcomeScreen from "./components/WelcomeScreen.jsx";
import HRAdminPanel from "./components/HRAdminPanel.jsx";
import UnauthorizedPage from "./components/UnauthorizedPage.jsx";
import Sidebar from "./components/Sidebar.jsx";
import ProfileModal from "./components/ProfileModal.jsx";
import ConnectionModal from "./components/ConnectionModal.jsx"; // global connections UI

import SuperAdminLayout from './pages/superadmin/SuperAdminLayout.jsx';
import SAUserList from './pages/superadmin/UserList.jsx';
import Overview from "./pages/superadmin/Overview.jsx";
import SAAddUser from './pages/superadmin/AddUser.jsx';
import SASubscriptions from './pages/superadmin/Subscriptions.jsx';
import SAStatuses from './pages/superadmin/Statuses.jsx';
import SATickets from './pages/superadmin/Tickets.jsx';
import SARevenue from './pages/superadmin/Revenue.jsx';
import StaffLogin from "./pages/staff/StaffLogin";
import Pricing from "./pages/Pricing.jsx";
import Workspace from "./components/Workspace.jsx";
import WorkspaceAccept from "./pages/WorkspaceAccept";

// Pages
import Login from "./pages/Login.jsx";
import Signup from "./pages/Signup.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import TrialEnded from "./pages/TrialEnded.jsx";

/* ───────────────────────── helpers ───────────────────────── */
async function apiFetch(url, options = {}) {
  const res = await fetch(url, {
    credentials: "include",
    cache: "no-store",
    headers: { "Cache-Control": "no-cache", ...(options.headers || {}) },
    ...options,
  });
  if (res.status === 402 && window.location.pathname !== "/trial-ended") {
    window.location.replace("/trial-ended");
    throw new Error("trial_expired");
  }
  return res;
}

/* NEW: Guard that checks staff session via /api/admin/staff/me */
function StaffProtectedRoute({ children }) {
  const [state, setState] = React.useState({ loading: true, ok: false });

  React.useEffect(() => {
    (async () => {
      try {
        const r = await fetch("/api/admin/staff/me", {
          credentials: "include",
          cache: "no-store",
          headers: { "Cache-Control": "no-cache" },
        });
        const j = await r.json();
        setState({ loading: false, ok: !!j?.is_staff });
      } catch {
        setState({ loading: false, ok: false });
      }
    })();
  }, []);

  if (state.loading) {
    return (
      <div className="flex h-screen items-center justify-center text-white bg-[#0b1324]">
        Checking staff access…
      </div>
    );
  }
  return state.ok ? children : <Navigate to="/staff/login" replace />;
}

async function isTrialExpired() {
  try {
    const r = await fetch("/api/trial/status", {
      credentials: "include",
      cache: "no-store",
      headers: { "Cache-Control": "no-cache" },
    });
    const j = await r.json();
    return !!j?.expired;
  } catch {
    return false;
  }
}

function ProtectedRoute({ children }) {
  const [state, setState] = React.useState({ loading: true, ok: false });

  React.useEffect(() => {
    (async () => {
      const expired = await isTrialExpired();
      if (expired) {
        if (window.location.pathname !== "/trial-ended") {
          window.location.replace("/trial-ended");
        }
        return;
      }
      try {
        const r = await fetch("/api/auth/me", {
          credentials: "include",
          cache: "no-store",
          headers: { "Cache-Control": "no-cache" },
        });
        const d = await r.json();
        setState({ loading: false, ok: !!d?.authenticated });
      } catch {
        setState({ loading: false, ok: false });
      }
    })();
  }, []);

  if (state.loading) {
    return (
      <div className="flex h-screen items-center justify-center text-white bg-[#0b1324]">
        Loading…
      </div>
    );
  }
  return state.ok ? children : <Navigate to="/login" replace />;
}

/* ───────────────────────── top bar ───────────────────────── */
function TopBar({
  onToggleSidebar,
  onOpenProfile,
  me,
  scope,
  setScope,
  connections,
  onActiveSourcesChange,
}) {
  const avatarUrl = me?.avatar_url || "";
  return (
    <header className="sticky top-0 z-20 flex items-center justify-between border-b border-white/10 bg-[#0b1324]/80 backdrop-blur px-3 py-2">
      <div className="flex items-center gap-2">
        <button
          type="button"
          className="md:hidden rounded-lg p-2 text-white/80 hover:text-white hover:bg-white/5"
          aria-label="Toggle sidebar"
          onClick={onToggleSidebar}
        >
          <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
            <path d="M3 6h18v2H3zM3 11h18v2H3zM3 16h18v2H3z" />
          </svg>
        </button>

        <HeaderModeMenu
          scope={scope}
          setScope={setScope}
          connections={connections}
          onActiveSourcesChange={onActiveSourcesChange}
        />
      </div>

      <div className="text-white/60 text-sm">ECHO · File Assistant</div>

      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onOpenProfile}
          className="grid h-8 w-8 place-items-center overflow-hidden rounded-full border border-white/10 bg-white/10 text-white"
          title="Profile"
        >
          {avatarUrl ? (
            <img src={avatarUrl} alt="avatar" className="h-full w-full object-cover" />
          ) : (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 12c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm0 2c-3.33 0-10 1.67-10 5v3h20v-3c0-3.33-6.67-5-10-5z" />
            </svg>
          )}
        </button>
      </div>
    </header>
  );
}

/* ───────────────────────── main app ───────────────────────── */
export default function App() {
  // Chat/session state
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isInitializing, setIsInitializing] = useState(true);

  const [pauseGPT, setPauseGPT] = useState(false);
  const [fileOptions, setFileOptions] = useState([]);
  const [allFiles, setAllFiles] = useState([]);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [showWelcome, setShowWelcome] = useState(true);

  const [userInput, setUserInput] = useState("");
  const [aiStatusMessage, setAiStatusMessage] = useState("");
  const [statusType, setStatusType] = useState("thinking");

  const [chatId, setChatId] = useState(null);
  const [refreshChats, setRefreshChats] = useState(false);
  const [userEmail, setUserEmail] = useState(null);

  const [page, setPage] = useState(1);
  const [totalFiles, setTotalFiles] = useState(0);
  const [fileType, setFileType] = useState("");

  const aiThinkingInterval = useRef(null);
  const [availableFileTypes, setAvailableFileTypes] = useState([]);
  const [me, setMe] = useState(null);
  const [connections, setConnections] = useState([]);

  // UI state
  const [sidebarOpen, setSidebarOpen] = useState(false); // mobile overlay
  const [profileOpen, setProfileOpen] = useState(false);

  // Search scope
  const [scope, setScope] = useState("all");              // "all" or one source key
  const [activeSources, setActiveSources] = useState([]); // used when scope === "all"

  const sourcesForScope = useMemo(
    () => (scope === "all" ? activeSources : [scope]),
    [scope, activeSources]
  );

  // Global connection modal
  const [connOpen, setConnOpen] = useState(false);
  const [connInitial, setConnInitial] = useState("home");
  useEffect(() => {
    const onOpen = (e) => {
      setConnInitial(e?.detail?.initial || "home");
      setConnOpen(true);
    };
    window.addEventListener("connections:open", onOpen);
    return () => window.removeEventListener("connections:open", onOpen);
  }, []);

  useEffect(() => {
    async function init() {
      try {
        if (window.location.pathname === "/trial-ended") {
          setIsInitializing(false);
          return;
        }

        // who am I
        const loginRes = await apiFetch("/api/auth/me");
        const loginData = await loginRes.json();
        if (!loginData?.authenticated) {
          setIsInitializing(false);
          return;
        }
        setUserEmail(loginData?.user?.email ?? loginData?.user_email ?? null);

        // restore chat
        const storedChatId = sessionStorage.getItem("chat_id");
        if (storedChatId) {
          const msgRes = await apiFetch(`/api/messages/${storedChatId}`);
          const msgData = await msgRes.json();
          if (msgData.messages && msgData.messages.length > 0) {
            setChatId(storedChatId);
            setMessages(msgData.messages);
            setShowWelcome(false);
          } else {
            sessionStorage.removeItem("chat_id");
            setShowWelcome(true);
          }
        } else {
          setShowWelcome(true);
        }

        // pending selection?
        const sessionRes = await apiFetch("/api/session_state");
        const sessionState = await sessionRes.json();
        if (sessionState.stage === "awaiting_selection" && sessionState.files?.length > 0) {
          setPauseGPT(true);
          setAllFiles(sessionState.files);
          setFileOptions(sessionState.files.slice(0, 5));
          setTotalFiles(sessionState.files.length);
          setPage(1);
        } else {
          setPauseGPT(false);
          setFileOptions([]);
          setAllFiles([]);
          setSelectedFiles([]);
          setStatusType("thinking");
          setAiStatusMessage("");
        }

        // profile + connections
        const profileRes = await apiFetch("/api/profile");
        setMe(await profileRes.json());
        try {
          const connRes = await apiFetch("/api/connections");
          const connData = await connRes.json();
          setConnections(connData?.connections || []);
        } catch { /* optional */ }
      } catch (err) {
        console.error("❌ Failed to initialize session:", err);
        setAiStatusMessage("❌ Something went wrong.");
        setStatusType("error");
      } finally {
        setIsInitializing(false);
      }
    }
    init();

    // refresh profile after ProfileModal changes
    const onProfileChanged = () => {
      apiFetch("/api/profile").then(r => r.json()).then(setMe).catch(() => {});
    };
    window.addEventListener("profile:changed", onProfileChanged);
    return () => window.removeEventListener("profile:changed", onProfileChanged);
  }, []);

  // New chat
  const handleNewChat = async () => {
    const res = await apiFetch("/api/new_chat");
    const data = await res.json();

    setChatId(data.chat_id);
    sessionStorage.removeItem("chat_id");

    setMessages([]);
    setPauseGPT(false);
    setSelectedFiles([]);
    setFileOptions([]);
    setAllFiles([]);
    setUserInput("");
    setShowWelcome(true);
    setAiStatusMessage("");
    setStatusType("thinking");
    clearInterval(aiThinkingInterval.current);
    setRefreshChats((prev) => !prev);
  };

  // Load existing chat
  const handleSelectChat = async (chat_id) => {
    if (chat_id === chatId) return;
    setChatId(chat_id);
    sessionStorage.setItem("chat_id", chat_id);
    setShowWelcome(false);
    setIsLoading(true);
    try {
      const res = await apiFetch(`/api/messages/${chat_id}`);
      const data = await res.json();
      setMessages(data.messages || []);
      setPauseGPT(false);
      setSelectedFiles([]);
      setFileOptions([]);
      setAllFiles([]);
      setUserInput("");
      setIsLoading(false);
    } catch (err) {
      console.error("❌ Failed to load chat messages:", err);
      setIsLoading(false);
    }
  };

  /**
   * MAIN SEND
   * onSend can pass an extra object: { attachments: UploadedFile[] }
   * We render those files in the user bubble and include their IDs in /chat.
   */
  const handleSend = async (
  inputText = userInput,
  selectedIndices = null,
  sourcesArg = [],
  extra = {}
) => {
  const msg = (inputText || "").trim();

  // nothing to do?
  const userAttachments = Array.isArray(extra.attachments) ? extra.attachments : [];
  if (!msg && !selectedIndices && !userAttachments.length) return;

  // ---- Push user bubble (keep chips visible after submit)
  const userBubble = { sender: "You", message: msg };
  if (userAttachments.length) userBubble.attachments = userAttachments.map(a => ({ ...a })); // clone
  if (msg || userAttachments.length) {
    setMessages(prev => [...prev, userBubble]);
  }

  setUserInput("");
  setShowWelcome(false);
  setIsLoading(true);

  // Status indicator
  if (pauseGPT || selectedIndices) {
    setAiStatusMessage("ECHO is checking access...");
    setStatusType("checking-access");
  } else if (msg.toLowerCase().includes("file") || userAttachments.length) {
    setAiStatusMessage("ECHO is searching the file...");
    setStatusType("searching-file");
  } else {
    setAiStatusMessage("ECHO is thinking...");
    setStatusType("thinking");
  }

  try {
    // ---- Build payload
    const payload = {
      message: msg || "",
      selectionStage: Boolean(pauseGPT || selectedIndices),
      selectedIndices,
      chat_id: chatId,
    };
    if (Array.isArray(sourcesArg)) payload.sources = sourcesArg;

    // NEW: include uploads in BOTH formats (for new + legacy backends)
    if (userAttachments.length) {
      payload.upload_ids = userAttachments.map(f => f.id).filter(Boolean);
      payload.attachments = userAttachments.map(f => ({
        id: f.id,
        name: f.name,
        path: f.path,
        url: f.url,
        mime: f.mime,
      }));
    }

    // NEW: if we are in selection stage but the UI is sending concrete files,
    // also pass `selectedFileIds` so the server coerces selection reliably.
    if (extra.selectedFileIds?.length) {
      payload.selectedFileIds = extra.selectedFileIds.slice(0);
    }

    const res = await apiFetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const isJson = (res.headers.get("content-type") || "").includes("application/json");
    const data = isJson ? await res.json() : { error: await res.text() };
    if (!res.ok) throw new Error(data?.error || `Request failed (${res.status})`);

    // ---- Assistant message (if any)
    if (data.response) {
      setMessages(prev => [...prev, { sender: "AI", message: data.response }]);
    }
    if (data.chat_id) setChatId(data.chat_id);

    // ---- File-selection (picker) flow
    if (data.pauseGPT && Array.isArray(data.files)) {
      setPauseGPT(true);
      setAllFiles(data.files);
      setFileOptions(data.files);
      setPage(data.page || 1);
      setTotalFiles(data.total || data.files.length);
      setAvailableFileTypes(data.file_types || []);
    } else {
      // reset picker state
      setPauseGPT(false);
      setFileOptions([]);
      setAllFiles([]);
      setSelectedFiles([]);
      setPage(1);
      setTotalFiles(0);
    }

    setIsLoading(false);
    if (aiThinkingInterval.current) clearInterval(aiThinkingInterval.current);
    setAiStatusMessage("");
    setRefreshChats(prev => !prev);
  } catch (err) {
    console.error("❌ Chat fetch failed:", err);
    setIsLoading(false);
    if (aiThinkingInterval.current) clearInterval(aiThinkingInterval.current);
    setAiStatusMessage("❌ Something went wrong.");
    setStatusType("error");
  }
};


  // Send selected files from the picker
  const sendSelectedFiles = () => {
    if (!selectedFiles.length) return;
    const selectedIndices = selectedFiles
      .map((file) => allFiles.findIndex((f) => f.id === file.id))
      .filter((i) => i !== -1)
      .map((i) => i + 1);
    handleSend(`Selected: ${selectedIndices.join(",")}`, selectedIndices);
    setSelectedFiles([]);
    setPauseGPT(false);
    setFileOptions([]);
  };

  const handlePageChange = async (newPage) => {
    if (newPage < 1 || newPage > Math.ceil(totalFiles / 5)) return;
    try {
      const res = await apiFetch(`/api/paginate_files?page=${newPage}&type=${fileType}`);
      const data = await res.json();
      if (data.files) {
        setFileOptions(data.files);
        setPage(data.page);
        setTotalFiles(data.total);
        setAvailableFileTypes(data.file_types || []);
        setAllFiles((prev) => {
          const seen = new Set(prev.map((f) => f.id));
          const merged = [...prev];
          for (const f of data.files) if (!seen.has(f.id)) merged.push(f);
          return merged;
        });
      }
    } catch (err) {
      console.error("❌ Failed to paginate files:", err);
    }
  };

  const handleFilterChange = async (newType) => {
    setFileType(newType);
    setPage(1);
    try {
      const res = await apiFetch(`/api/paginate_files?page=1&type=${newType}`);
      const data = await res.json();
      if (data.files) {
        setFileOptions(data.files);
        setTotalFiles(data.total);
        setAvailableFileTypes(data.file_types || []);
        setAllFiles(data.files);
      }
    } catch {}
  };

  const toggleSelectFile = (file) => {
    setSelectedFiles((prev) =>
      prev.some((f) => f.id === file.id) ? prev.filter((f) => f.id !== file.id) : [...prev, file]
    );
  };

  const skipFileSelection = async () => {
    try {
      await apiFetch("/api/skip_selection", { method: "POST" });
      setPauseGPT(false);
      setFileOptions([]);
      setSelectedFiles([]);
    } catch (err) {
      console.error("❌ Failed to skip selection:", err);
    }
  };

  // Summarize selected files (server endpoint must exist)
  const summarizeSelectedFiles = async (files) => {
    const body = { selectedIds: (files || []).map((f) => f.id) };
    const res = await fetch("/api/summarize_selected", {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();

    setMessages((prev) => [...prev, { sender: "AI", message: data.response }]);

    setPauseGPT(false);
    setSelectedFiles([]);
    setFileOptions([]);
  };

  // remember chat_id locally when we see the first user+ai pair
  useEffect(() => {
    const hasUser = messages.some((m) => m.sender === "You");
    const hasAI = messages.some((m) => m.sender === "AI");
    if (hasUser && hasAI && chatId && !sessionStorage.getItem("chat_id")) {
      sessionStorage.setItem("chat_id", chatId);
    }
  }, [messages, chatId]);

  if (isInitializing) {
    return (
      <div className="flex h-screen items-center justify-center text-white bg-[#0b1324]">
        Loading…
      </div>
    );
  }

  /* ───────────────────────── UI ───────────────────────── */
  return (
    <Router>
      <Routes>
              {/* Staff login page */}
        <Route path="/staff/login" element={<StaffLogin />} />
        <Route path="/pricing" element={<Pricing />} />
        <Route
          path="/workspace"
          element={
            <ProtectedRoute>
              <Workspace />
            </ProtectedRoute>
          }
        />
        <Route path="/workspace/accept" element={<WorkspaceAccept />} />

        {/* ⬇️ CHANGE: wrap superadmin with StaffProtectedRoute (not ProtectedRoute) */}
        <Route
          path="/superadmin"
          element={
            <StaffProtectedRoute>
              <SuperAdminLayout />
            </StaffProtectedRoute>
          }
        >
          <Route index element={<SAUserList />} />
          <Route path="overview" element={<Overview />} />
          <Route path="userlist" element={<SAUserList />} />
          <Route path="add-user" element={<SAAddUser />} />
          <Route path="subscriptions" element={<SASubscriptions />} />
          <Route path="statuses" element={<SAStatuses />} />
          <Route path="tickets" element={<SATickets />} />
          <Route path="revenue" element={<SARevenue />} />
        </Route>
        {/* Public */}
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/unauthorized" element={<UnauthorizedPage />} />
        <Route path="/trial-ended" element={<TrialEnded />} />

        {/* Protected */}
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          }
        />

        <Route
          path="/admin/upload"
          element={
            <ProtectedRoute>
              <div className="flex h-screen bg-[#0b1324] text-white p-6">
                <HRAdminPanel userEmail={userEmail} />
              </div>
            </ProtectedRoute>
          }
        />

        {/* Main assistant */}
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <div className="flex h-screen overflow-hidden bg-[#0b1324] text-white">
                {/* Sidebar (desktop) */}
                <div className="hidden md:block w-[300px] shrink-0 border-r border-white/10 bg-[#0b1324] sticky top-0 h-screen">
                  <Sidebar
                    onNewChat={handleNewChat}
                    onSelectChat={handleSelectChat}
                    activeChatId={chatId}
                    refreshFlag={refreshChats}
                    onOpenProfile={() => setProfileOpen(true)}
                    onLogout={() => fetch("/api/auth/logout").then(() => (location.href = "/login"))}
                    selectedScope={scope}
                    onScopeChange={setScope}
                    onActiveSourcesChange={setActiveSources}
                  />
                </div>

                {/* Sidebar (mobile overlay) */}
                {sidebarOpen && (
                  <div className="md:hidden fixed inset-0 z-40 flex">
                    <div className="w-[84%] max-w-[320px] border-r border-white/10 bg-[#0b1324]">
                      <Sidebar
                        onNewChat={handleNewChat}
                        onSelectChat={handleSelectChat}
                        activeChatId={chatId}
                        refreshFlag={refreshChats}
                        onOpenProfile={() => setProfileOpen(true)}
                        onLogout={() => fetch("/api/auth/logout").then(() => (location.href = "/login"))}
                        selectedScope={scope}
                        onScopeChange={setScope}
                        onActiveSourcesChange={setActiveSources}
                      />
                    </div>
                    <div className="flex-1 bg-black/50" onClick={() => setSidebarOpen(false)} />
                  </div>
                )}

                {/* Main column */}
                <div className="flex-1 flex flex-col overflow-hidden">
                  <TopBar
                    onToggleSidebar={() => setSidebarOpen((s) => !s)}
                    onOpenProfile={() => setProfileOpen(true)}
                    me={me}
                    scope={scope}
                    setScope={setScope}
                    connections={connections}
                    onActiveSourcesChange={setActiveSources}
                  />

                  {showWelcome ? (
                    <WelcomeScreen
                      userInput={userInput}
                      setUserInput={setUserInput}
                      onSend={(msg) => handleSend(msg, null, sourcesForScope)}
                      onNewChat={handleNewChat}
                    />
                  ) : (
                    <>
                      {/* Conversation */}
                      <div className="flex-1 overflow-y-auto px-4 pb-28 custom-scrollbar">
                        <div className="mx-auto w-full max-w-[980px]">
                          <ChatPanel
                            messages={
                              isLoading && aiStatusMessage
                                ? [
                                    ...messages,
                                    { sender: "AI", message: aiStatusMessage, isStatus: true, statusType },
                                  ]
                                : messages
                            }
                            fileOptions={fileOptions}
                            allFiles={allFiles}
                            pauseGPT={pauseGPT}
                            toggleSelectFile={toggleSelectFile}
                            selectedFiles={selectedFiles}
                            sendSelectedFiles={sendSelectedFiles}
                            summarizeSelectedFiles={() => summarizeSelectedFiles(selectedFiles)}
                            onUserMessage={(text) =>
                              setMessages((m) => [...m, { sender: "You", message: text }])
                            }
                            page={page}
                            totalFiles={totalFiles}
                            onPageChange={handlePageChange}
                            onFilterChange={handleFilterChange}
                            onSkipFileSelection={skipFileSelection}
                            availableFileTypes={availableFileTypes}
                            onSelectChat={handleSelectChat}
                            onNewChat={handleNewChat}
                            userAvatarUrl={me?.avatar_url}
                          />
                        </div>
                      </div>

                      {/* Composer */}
                      <div className="sticky bottom-0 left-0 right-0">
                        <div className="mx-auto w-full max-w-[980px]">
                          <ChatInput
                            onSend={(msg, selectedIndices, sources, extra) =>
                              handleSend(msg, selectedIndices ?? null, sources ?? sourcesForScope, extra)
                            }
                            disabled={isLoading}
                            userInput={userInput}
                            setUserInput={setUserInput}
                          />
                        </div>
                      </div>
                    </>
                  )}
                </div>

                {profileOpen && (
                  <ProfileModal
                    me={me}
                    connections={connections}
                    onClose={() => setProfileOpen(false)}
                  />
                )}
              </div>
            </ProtectedRoute>
          }
        />

        {/* Fallbacks */}
        <Route path="/admin" element={<Navigate to="/admin/upload" replace />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>

      {/* Global connection modal host */}
      {connOpen && (
        <ConnectionModal
          open={connOpen}
          initial={connInitial}
          onClose={() => setConnOpen(false)}
        />
      )}
    </Router>
  );
}
