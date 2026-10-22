"""Atomically establish a protected zero before setup or after verifier failure."""
import os
import secrets
import stat

if os.geteuid() != 0:
    raise RuntimeError("Verifier bootstrap requires root")

flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
logs = os.open('/logs', flags)
try:
    os.fchown(logs, 0, 0)
    os.fchmod(logs, 0o755)
    try:
        entry = os.stat('verifier', dir_fd=logs, follow_symlinks=False)
        if not stat.S_ISDIR(entry.st_mode):
            os.unlink('verifier', dir_fd=logs)
    except FileNotFoundError:
        pass
    try:
        os.mkdir('verifier', mode=0o700, dir_fd=logs)
    except FileExistsError:
        pass
    output = os.open('verifier', flags, dir_fd=logs)
    try:
        os.fchown(output, 0, 0)
        os.fchmod(output, 0o700)
        # Rename any old entry, including directories and symlinks, without
        # traversing it. The protected directory prevents candidate races.
        # Clear the alternative reward format before installing the zero.
        for old in ('reward.json', 'reward.txt'):
            try:
                os.rename(old, '.quarantined-' + secrets.token_hex(16),
                          src_dir_fd=output, dst_dir_fd=output)
            except FileNotFoundError:
                pass
        name = '.zero-' + secrets.token_hex(16)
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=output)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(b'0\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, 'reward.txt', src_dir_fd=output, dst_dir_fd=output)
    finally:
        os.close(output)
finally:
    os.close(logs)
