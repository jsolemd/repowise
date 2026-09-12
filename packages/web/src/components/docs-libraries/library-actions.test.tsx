// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ callDocsTool: vi.fn() }));
vi.mock("@/lib/api/docs-libraries", () => ({ callDocsTool: mocks.callDocsTool }));

import { AddDocLibrary } from "./library-actions";

beforeEach(() => vi.clearAllMocks());
afterEach(cleanup);

function openForm() {
  fireEvent.click(screen.getByRole("button", { name: "Add library" }));
  fireEvent.change(screen.getByLabelText("Library name"), { target: { value: " Example " } });
  fireEvent.change(screen.getByLabelText("GitHub repository"), { target: { value: " owner/example " } });
}

it("reports existing libraries without claiming that a new job was queued", async () => {
  mocks.callDocsTool.mockResolvedValue({ disposition: "exists", library_name: "Example", current_status: "ready" });
  const onAdded = vi.fn();
  render(<AddDocLibrary onAdded={onAdded} />);
  openForm();
  fireEvent.click(screen.getByRole("button", { name: "Add and index" }));

  expect(await screen.findByText("Example is already registered (ready).")).toBeTruthy();
  expect(screen.queryByText(/indexing is queued/i)).toBeNull();
  expect(onAdded).toHaveBeenCalledOnce();
  expect(document.activeElement).toBe(screen.getByRole("button", { name: "Add library" }));
});

it("submits a selected branch and trims optional paths", async () => {
  mocks.callDocsTool.mockResolvedValue({ disposition: "created", library_name: "Example" });
  render(<AddDocLibrary onAdded={vi.fn()} />);
  openForm();
  const toggle = screen.getByRole("button", { name: "Add library" });
  expect(toggle.getAttribute("aria-expanded")).toBe("true");
  expect(document.getElementById(toggle.getAttribute("aria-controls")!)).toBeTruthy();
  fireEvent.change(screen.getByLabelText("Branch"), { target: { value: " master " } });
  fireEvent.change(screen.getByLabelText("Docs directory (optional)"), { target: { value: " docs/api " } });
  fireEvent.click(screen.getByRole("button", { name: "Add and index" }));

  await waitFor(() => expect(mocks.callDocsTool).toHaveBeenCalledWith("add_doc_library", {
    repo: "owner/example", name: "Example", branch: "master", docs_path: "docs/api",
  }));
  expect(await screen.findByText("Library added. Indexing is queued.")).toBeTruthy();
});

it("keeps a failed registration editable without reporting success", async () => {
  mocks.callDocsTool.mockRejectedValue(new Error("Branch not found"));
  const onAdded = vi.fn();
  render(<AddDocLibrary onAdded={onAdded} />);
  openForm();
  fireEvent.click(screen.getByRole("button", { name: "Add and index" }));

  expect(await screen.findByText("Branch not found")).toBeTruthy();
  expect((screen.getByLabelText("Library name") as HTMLInputElement).value).toBe(" Example ");
  expect((screen.getByRole("button", { name: "Add and index" }) as HTMLButtonElement).disabled).toBe(false);
  expect(onAdded).not.toHaveBeenCalled();
});
