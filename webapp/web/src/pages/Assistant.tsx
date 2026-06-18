/**
 * webapp/web/src/pages/Assistant.tsx
 * ------------------------------------
 * Conversational assistant page — persists chat history, renders inline
 * PersonCard results, and surfaces Confirm / Cancel for proposed write
 * actions (merge, delete, edit_attributes, redescribe).
 */

import { useEffect, useRef, useState } from "react";
import {
  Bot,
  CheckCircle,
  ChevronRight,
  Loader2,
  MessageCircle,
  RotateCcw,
  Send,
  XCircle,
} from "lucide-react";
import { PersonCard } from "../components/PersonCard";
import { PersonDrawer } from "../components/PersonDrawer";
import type { ChatReply } from "../api/client";
import {
  getChatMessages,
  postChat,
} from "../api/client";
import type { PersonSummary } from "../api/types";

// ── Types ─────────────────────────────────────────────────────────────────────
interface Bubble {
  id: string;
  role: "user" | "assistant";
  content: string;
  results?: PersonSummary[] | null;
  open_person_id?: string | null;
  proposed_action?: ChatReply["proposed_action"] | null;
  loading?: boolean;
}

const SESSION_KEY = "surv_chat_session";

function uid() {
  return Math.random().toString(36).slice(2, 10);
}

// ── Markdown-lite renderer (bold + bullet-list only) ──────────────────────────
function renderMd(text: string) {
  const lines = text.split("\n");
  return lines.map((line, i) => {
    // bold
    const parts = line.split(/(\*\*[^*]+\*\*)/g).map((s, j) =>
      s.startsWith("**") ? (
        <strong key={j} className="text-emerald-glow">
          {s.slice(2, -2)}
        </strong>
      ) : (
        s
      ),
    );
    if (line.startsWith("•")) {
      return (
        <div key={i} className="flex gap-2 leading-relaxed">
          <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-emerald/60" />
          <span>{parts.slice(1)}</span>
        </div>
      );
    }
    return (
      <p key={i} className="leading-relaxed">
        {parts}
      </p>
    );
  });
}

// ── Single bubble ─────────────────────────────────────────────────────────────
function BotBubble({
  bubble,
  onConfirm,
  onCancel,
  onPersonClick,
}: {
  bubble: Bubble;
  onConfirm: () => void;
  onCancel: () => void;
  onPersonClick: (id: string) => void;
}) {
  return (
    <div className="group flex gap-3">
      {/* Avatar */}
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-emerald/10 text-emerald-glow ring-1 ring-emerald/20">
        <Bot size={16} />
      </div>

      <div className="flex-1 space-y-3">
        {/* Text */}
        <div className="rounded-2xl rounded-tl-sm bg-ink-700/80 px-4 py-3 text-sm text-emerald-100/90 shadow ring-1 ring-white/5">
          {bubble.loading ? (
            <Loader2 size={16} className="animate-spin text-emerald/60" />
          ) : (
            <div className="space-y-1">{renderMd(bubble.content)}</div>
          )}
        </div>

        {/* Results grid */}
        {bubble.results && bubble.results.length > 0 && (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
            {bubble.results.map((p) => (
              <PersonCard
                key={p.person_id}
                p={p}
                onClick={() => onPersonClick(p.person_id)}
              />
            ))}
          </div>
        )}

        {/* Proposed action */}
        {bubble.proposed_action && (
          <div className="flex items-center gap-3 rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
            <ChevronRight size={16} className="shrink-0 text-amber-400" />
            <span className="flex-1">{bubble.proposed_action.summary}</span>
            <button
              id="chat-confirm-btn"
              onClick={onConfirm}
              className="flex items-center gap-1 rounded-lg bg-emerald/20 px-3 py-1 text-xs font-medium text-emerald-glow transition hover:bg-emerald/30"
            >
              <CheckCircle size={14} /> Confirm
            </button>
            <button
              id="chat-cancel-btn"
              onClick={onCancel}
              className="flex items-center gap-1 rounded-lg bg-red-500/15 px-3 py-1 text-xs font-medium text-red-300 transition hover:bg-red-500/25"
            >
              <XCircle size={14} /> Cancel
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function UserBubble({ content }: { content: string }) {
  return (
    <div className="flex justify-end gap-3">
      <div className="max-w-lg rounded-2xl rounded-tr-sm bg-emerald/15 px-4 py-3 text-sm text-emerald-100 shadow ring-1 ring-emerald/20">
        {content}
      </div>
    </div>
  );
}

// ── Suggested starters ────────────────────────────────────────────────────────
const SUGGESTIONS = [
  "Find a child about 9 wearing a red t-shirt",
  "How many people are tracked?",
  "Show confirmed people on camera 2",
  "Any violence alerts?",
];

// ── Main page ─────────────────────────────────────────────────────────────────
export default function Assistant() {
  const [bubbles, setBubbles] = useState<Bubble[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [sessionId, setSessionId] = useState<string | undefined>(
    () => sessionStorage.getItem(SESSION_KEY) || undefined,
  );
  const [drawerPid, setDrawerPid] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Load previous messages on first mount
  useEffect(() => {
    if (!sessionId) return;
    getChatMessages(sessionId)
      .then(({ messages }) => {
        const loaded: Bubble[] = messages.map((m) => ({
          id: uid(),
          role: m.role,
          content: m.content,
        }));
        if (loaded.length) setBubbles(loaded);
      })
      .catch(() => {
        // Session may have expired — reset
        sessionStorage.removeItem(SESSION_KEY);
        setSessionId(undefined);
      });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Scroll to bottom whenever bubbles change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [bubbles]);

  const addUserBubble = (text: string) =>
    setBubbles((prev) => [...prev, { id: uid(), role: "user", content: text }]);

  const addLoadingBubble = (): string => {
    const id = uid();
    setBubbles((prev) => [
      ...prev,
      { id, role: "assistant", content: "", loading: true },
    ]);
    return id;
  };

  const replaceBubble = (id: string, data: Partial<Bubble>) =>
    setBubbles((prev) =>
      prev.map((b) => (b.id === id ? { ...b, ...data, loading: false } : b)),
    );

  const send = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || sending) return;
    setInput("");
    setSending(true);
    addUserBubble(trimmed);
    const loadId = addLoadingBubble();
    try {
      const reply = await postChat(trimmed, sessionId);
      if (reply.session_id !== sessionId) {
        setSessionId(reply.session_id);
        sessionStorage.setItem(SESSION_KEY, reply.session_id);
      }
      replaceBubble(loadId, {
        content: reply.reply,
        results: reply.results,
        open_person_id: reply.open_person_id,
        proposed_action: reply.proposed_action,
      });
      // Auto-open person if the brain resolved a lookup
      if (reply.open_person_id) setDrawerPid(reply.open_person_id);
    } catch {
      replaceBubble(loadId, {
        content: "⚠️ Couldn't reach the server. Is the API running?",
      });
    } finally {
      setSending(false);
    }
  };

  const handleConfirm = async (bubble: Bubble) => {
    // Clear the proposed action from the bubble immediately
    setBubbles((prev) =>
      prev.map((b) =>
        b.id === bubble.id ? { ...b, proposed_action: null } : b,
      ),
    );
    await send("yes");
  };

  const handleCancel = async (bubble: Bubble) => {
    setBubbles((prev) =>
      prev.map((b) =>
        b.id === bubble.id ? { ...b, proposed_action: null } : b,
      ),
    );
    await send("no");
  };

  const clearSession = () => {
    sessionStorage.removeItem(SESSION_KEY);
    setSessionId(undefined);
    setBubbles([]);
  };

  const isEmpty = bubbles.length === 0;

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald/10 text-emerald-glow ring-1 ring-emerald/20">
            <MessageCircle size={20} />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-emerald-glow">
              Assistant
            </h1>
            <p className="text-xs text-emerald-200/50">
              Ask me to find people, run stats, or make corrections
            </p>
          </div>
        </div>
        {!isEmpty && (
          <button
            id="chat-clear-btn"
            onClick={clearSession}
            className="btn-ghost flex items-center gap-2 text-xs"
            title="Start a new conversation"
          >
            <RotateCcw size={14} />
            New chat
          </button>
        )}
      </div>

      {/* Chat area */}
      <div className="relative flex-1 overflow-hidden rounded-2xl bg-ink-800/40 ring-1 ring-white/5">
        <div className="h-full overflow-y-auto px-6 py-6">
          {isEmpty ? (
            /* Empty state */
            <div className="flex h-full flex-col items-center justify-center gap-8 text-center">
              <div>
                <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-emerald/10 ring-1 ring-emerald/20">
                  <Bot size={32} className="text-emerald-glow" />
                </div>
                <h2 className="text-xl font-semibold text-emerald-100">
                  SURVEILLANT Assistant
                </h2>
                <p className="mt-2 max-w-sm text-sm text-emerald-200/50">
                  I can search for people by description, open person details,
                  report stats, and — with your confirmation — merge, delete, or
                  correct identities.
                </p>
              </div>
              {/* Suggestion chips */}
              <div className="flex flex-wrap justify-center gap-2">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => send(s)}
                    className="rounded-full border border-emerald/20 bg-emerald/5 px-4 py-2 text-sm text-emerald-200/80 transition hover:border-emerald/40 hover:bg-emerald/10 hover:text-emerald-glow"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-5">
              {bubbles.map((b) =>
                b.role === "user" ? (
                  <UserBubble key={b.id} content={b.content} />
                ) : (
                  <BotBubble
                    key={b.id}
                    bubble={b}
                    onConfirm={() => handleConfirm(b)}
                    onCancel={() => handleCancel(b)}
                    onPersonClick={setDrawerPid}
                  />
                ),
              )}
              <div ref={bottomRef} />
            </div>
          )}
        </div>
      </div>

      {/* Input bar */}
      <form
        className="mt-4 flex items-end gap-3"
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
      >
        <div className="relative flex-1">
          <textarea
            id="chat-input"
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send(input);
              }
            }}
            placeholder={`Ask me anything — "find a man with a backpack on camera 3"…`}
            className="w-full resize-none rounded-xl border border-line bg-ink-700/60 px-4 py-3 pr-12 text-sm text-emerald-100 placeholder-emerald-200/30 outline-none transition focus:border-emerald/50 focus:ring-1 focus:ring-emerald/30"
            style={{ maxHeight: "8rem", overflowY: "auto" }}
          />
        </div>
        <button
          id="chat-send-btn"
          type="submit"
          disabled={!input.trim() || sending}
          className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-emerald/20 text-emerald-glow ring-1 ring-emerald/30 transition hover:bg-emerald/30 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {sending ? (
            <Loader2 size={18} className="animate-spin" />
          ) : (
            <Send size={18} />
          )}
        </button>
      </form>

      {/* Person drawer */}
      {drawerPid && (
        <PersonDrawer
          personId={drawerPid}
          onClose={() => setDrawerPid(null)}
        />
      )}
    </div>
  );
}
