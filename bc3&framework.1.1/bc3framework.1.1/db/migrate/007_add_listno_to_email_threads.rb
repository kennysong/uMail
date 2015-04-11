class AddListnoToEmailThreads < ActiveRecord::Migration
  def self.up
    add_column :email_threads, :listno, :string
  end

  def self.down
    remove_column :email_threads, :listno
  end
end
