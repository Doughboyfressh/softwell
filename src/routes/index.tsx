import { createFileRoute } from "@tanstack/react-router";
import { SoftwellApp } from "@/components/softwell-app";

export const Route = createFileRoute("/")({ component: Home });

function Home() {
  return <SoftwellApp />;
}
