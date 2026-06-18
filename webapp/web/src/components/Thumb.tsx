import { mediaUrl } from "../api/client";

export function Thumb({ url, className = "" }: { url?: string | null; className?: string }) {
  if (!url)
    return (
      <div className={`grid place-items-center bg-ink-600 text-2xl text-emerald-200/20 ${className}`}>
        ?
      </div>
    );
  return <img src={mediaUrl(url)} className={`object-cover ${className}`} loading="lazy" alt="" />;
}
