import * as React from "react";
import { describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { AdaptivePanel } from "../../src/shared/adaptive-panel.js";

function Harness({ modal = true, preventOutside = false, showOpener = true, onOutside }: {
  modal?: boolean;
  preventOutside?: boolean;
  showOpener?: boolean;
  onOutside?: () => void;
}) {
  const [open, setOpen] = React.useState(false);
  return (
    <>
      {showOpener && <button onClick={() => setOpen(true)}>Open panel</button>}
      <button>Outside action</button>
      <AdaptivePanel open={open} onOpenChange={setOpen} title="Details" modal={modal}
        onInteractOutside={(event) => {
          onOutside?.();
          if (preventOutside) event.preventDefault();
        }}>
        <button>Panel action</button>
      </AdaptivePanel>
    </>
  );
}

function openPanel() {
  const opener = screen.getByRole("button", { name: "Open panel" });
  act(() => opener.focus());
  fireEvent.click(opener);
  expect(screen.getByRole("dialog")).toContainElement(document.activeElement as HTMLElement);
  return opener;
}

describe("AdaptivePanel focus return", () => {
  it.each([true, false])("returns to its invoker after Escape and reopening (modal=%s)", async (modal) => {
    render(<Harness modal={modal} />);
    const opener = openPanel();
    fireEvent.keyDown(document.activeElement!, { key: "Escape" });
    await waitFor(() => expect(opener).toHaveFocus());
    openPanel();
    fireEvent.click(screen.getByRole("button", { name: "Close panel" }));
    await waitFor(() => expect(opener).toHaveFocus());
  });

  it("preserves focus deliberately moved outside a nonmodal panel", async () => {
    render(<Harness modal={false} />);
    openPanel();
    const outside = screen.getByRole("button", { name: "Outside action" });
    act(() => outside.focus());
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(outside).toHaveFocus();
  });

  it("returns focus when an outside interaction was prevented", async () => {
    const onOutside = vi.fn();
    render(<Harness modal={false} preventOutside onOutside={onOutside} />);
    const opener = openPanel();
    // Radix installs its outside-pointer listener on the next task.
    await act(() => new Promise((resolve) => setTimeout(resolve, 0)));
    fireEvent.pointerDown(screen.getByRole("button", { name: "Outside action" }), { pointerType: "mouse" });
    expect(onOutside).toHaveBeenCalledOnce();
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    fireEvent.keyDown(document.activeElement!, { key: "Escape" });
    await waitFor(() => expect(opener).toHaveFocus());
  });

  it("closes safely when the invoker was removed", async () => {
    const { rerender } = render(<Harness />);
    const opener = openPanel();
    rerender(<Harness showOpener={false} />);
    expect(opener.isConnected).toBe(false);
    fireEvent.keyDown(document.activeElement!, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(document.body).toHaveFocus();
  });

  it("returns from a nested panel to its parent invoker, then to the page", async () => {
    function Nested() {
      const [parent, setParent] = React.useState(false);
      const [child, setChild] = React.useState(false);
      return <>
        <button onClick={() => setParent(true)}>Open panel</button>
        <AdaptivePanel open={parent} onOpenChange={setParent} title="Parent">
          <button onClick={() => setChild(true)}>Open child</button>
          <AdaptivePanel open={child} onOpenChange={setChild} title="Child">Child details</AdaptivePanel>
        </AdaptivePanel>
      </>;
    }
    render(<Nested />);
    const opener = openPanel();
    const childOpener = screen.getByRole("button", { name: "Open child" });
    act(() => childOpener.focus());
    fireEvent.click(childOpener);
    fireEvent.keyDown(document.activeElement!, { key: "Escape" });
    await waitFor(() => expect(childOpener).toHaveFocus());
    fireEvent.keyDown(document.activeElement!, { key: "Escape" });
    await waitFor(() => expect(opener).toHaveFocus());
  });
});
