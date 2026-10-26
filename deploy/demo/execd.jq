# The sandbox's log lines, in the shape the demo's checks and report read.
#
# execd writes one text line per event, `2026-10-26T06:05:06.862Z INFO  <message>`, and the message carries
# `key=value` fields (values that come from outside are Debug-quoted). One per request, when it finishes:
#   request kind=exec|job|stream|callback command="bash" status=200 outcome="exit" ms=12
# and `execd <version> listening listen=… bearer=true run_as=true notify=true` once at startup. A callback
# line carries `job="…" attempt=1 status=Some(204)|None error=None|Some("…") ms=…` instead of a command.
#
# `execd_events` takes one raw log line (a string) and yields the events it stands for, each
# `{ts_ms, src:"sandbox", event, detail}`; for any line that is not an execd request or startup line it
# yields nothing, so a caller keeps such a line as it is.
#
#   listening           the startup line
#   exec_request        every command or job request answered, stamped when it ARRIVED (end - ms)
#   exec_start          a streamed command, likewise
#   exec_end            a streamed command, stamped when it finished
#   job_ended_callback  one attempt at a job-end callback: job, attempt, status, error, took_ms

def ts_ms:
  capture("^(?<s>[^.Z]+)(\\.(?<f>[0-9]+))?Z$")
  | (((.s + "Z") | fromdateiso8601) * 1000) + (((.f // "0") + "000")[0:3] | tonumber);

# One field value of a message: a quoted string, Some(x), None, a number, a boolean, or a bare word.
def val:
  if startswith("\"") then fromjson
  elif . == "None" then null
  elif startswith("Some(") then (.[5:-1] | val)
  elif test("^[0-9]+$") then tonumber
  elif . == "true" then true
  elif . == "false" then false
  else . end;
def fields:
  reduce (scan("(\\w+)=(Some\\(\"(?:[^\"\\\\]|\\\\.)*\"\\)|\"(?:[^\"\\\\]|\\\\.)*\"|\\S+)")) as $m ({}; .[$m[0]] = ($m[1] | val));

def sandbox($ts; $event; $detail): {ts_ms: $ts, src: "sandbox", event: $event, detail: $detail};

def execd_events:
  (capture("^(?<timestamp>[0-9]{4}-[0-9]{2}-[0-9]{2}T[^ ]+Z) +[A-Z]+ +(?<message>.*)$")? // null) as $l
  | if $l == null then empty
    elif ($l.message | test("^execd [^ ]+ listening ")) then
      ($l.message | fields) as $f
      | sandbox($l.timestamp | ts_ms; "listening"; {listen: $f.listen, bearer_required: $f.bearer, run_as: $f.run_as, notify: $f.notify})
    elif ($l.message | startswith("request ")) | not then empty
    else
      ($l.message | fields) as $f
      | ($l.timestamp | ts_ms) as $end
      | ($end - ($f.ms // 0)) as $start
      | if $f.kind == "callback" then
          sandbox($end; "job_ended_callback";
            {job: $f.job, attempt: $f.attempt, status: $f.status, error: $f.error, took_ms: $f.ms})
        elif $f.kind == "exec" or $f.kind == "job" or $f.kind == "stream" then
          sandbox($start; "exec_request"; {command: $f.command}),
          (if $f.kind == "stream" then
             sandbox($start; "exec_start"; {command: $f.command}),
             sandbox($end; "exec_end"; {command: $f.command})
           else empty end)
        else empty end
    end;
