import Foundation

/// Persists a security-scoped bookmark to the last loaded model directory so a
/// sandboxed app can re-open it on the next launch. App-sandbox bookmarks require
/// `.withSecurityScope`; resolving one yields a URL you must bracket with
/// `start/stopAccessingSecurityScopedResource()`.
enum ModelBookmark {
    private static let key = "RFDETRApp.lastModelBookmark"

    /// Store a bookmark to `url`. Must be called while `url` (or an enclosing
    /// security scope) is access-granted — e.g. just back from an open panel, or
    /// inside an active parent scope. Failure is non-fatal (no auto-restore next launch).
    static func store(_ url: URL) {
        do {
            let data = try url.bookmarkData(options: .withSecurityScope,
                                            includingResourceValuesForKeys: nil,
                                            relativeTo: nil)
            UserDefaults.standard.set(data, forKey: key)
        } catch {
            print("[ModelBookmark] store failed for \(url.path): \(error)")
        }
    }

    /// Resolve the stored bookmark, if any. Access is NOT started — the caller must
    /// bracket use with `start/stopAccessingSecurityScopedResource()`. Refreshes the
    /// bookmark if the OS reports it stale.
    static func resolve() -> URL? {
        guard let data = UserDefaults.standard.data(forKey: key) else { return nil }
        var isStale = false
        guard let url = try? URL(resolvingBookmarkData: data,
                                 options: .withSecurityScope,
                                 relativeTo: nil,
                                 bookmarkDataIsStale: &isStale) else { return nil }
        if isStale {
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            store(url)
        }
        return url
    }

    static func clear() { UserDefaults.standard.removeObject(forKey: key) }
}
